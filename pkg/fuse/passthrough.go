/*
 * JuiceFS, Copyright 2026 Juicedata, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package fuse

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/hanwen/go-fuse/v2/fuse"
	"github.com/juicedata/juicefs/pkg/meta"
	"github.com/juicedata/juicefs/pkg/utils"
	"github.com/juicedata/juicefs/pkg/vfs"
	"github.com/zeebo/blake3"
)

// passthroughState manages per-open backing files used for FUSE passthrough
// write acceleration. EXPERIMENTAL: scoped to the write path. A file opened
// for write gets a local staging file (on a non-stacked fs) that the kernel
// reads/writes directly via FUSE_PASSTHROUGH, bypassing the daemon per-op. On
// release the staging file is reconciled into JuiceFS slices via the normal
// writer path. Durability is therefore deferred to release (commit-style),
// with one exception: fsync(2)/fdatasync(2) reconcile immediately (see
// fsync below), so an application's explicit sync point means what it says.
//
// Backing registrations are NOT reused across opens: each write-open gets a
// fresh registration (a new ioctl or, on unprivileged broker mounts, a new
// RPC round trip to the node broker, plus a fresh staging-file create), and
// reconcile() unregisters and removes it — for a DIFFERENT inode's next open,
// never the same one. This was originally pooled (checked-in backings parked
// for reuse by whichever inode opened next), which is measurably cheaper for
// small-file-heavy workloads, but pooling handed a backing file across
// inodes: once mmap(2) has been called on a passthrough fd, the kernel's
// fuse_passthrough_mmap() repoints the VMA directly at the backing file
// (vma_set_file), fully decoupling the mapping's lifetime from the fd — a
// process can close(2) (triggering reconcile+recycle) while a still-live,
// still-dirty mapping keeps writing to what is now, after recycling, a
// DIFFERENT inode's staging file. There is no FUSE opcode or other userspace
// signal that observes mmap activity on a passthrough-diverted file (the
// entire point of passthrough is to bypass such upcalls), so there is no way
// to detect "this backing might still be mmap'd" and only pool the ones that
// aren't. Reusing a backing across inodes is a real, silent cross-tenant
// data-corruption vector; not reusing it is not.
type passthroughState struct {
	server *fuse.Server
	dir    string

	mu       sync.Mutex
	files    map[uint64]*ptFile // keyed by fh
	busy     map[Ino]int        // inodes with a passthrough open or pending reconcile
	seq      int                // monotonic counter for unique staging file names
	disabled bool               // registration failed with a permanent error; stop trying
	paused   bool               // draining for handover/shutdown; refuse new passthrough opens
	warnOne  sync.Once

	// Staging headroom. Passthrough writes land in a real file on this
	// filesystem, so an open granted without room for the data either fails
	// mid-write with ENOSPC or — when staging is a memory-backed emptyDir —
	// pushes the pod's memory cgroup into reclaim, which degrades every
	// subsequent staging write and reconcile. Cheaper to decline the open and
	// take the daemon path, which streams into the write-back cache instead.
	minFree      uint64
	freeCheckAt  time.Time
	freeOK       bool
	lowSpaceWarn sync.Once
}

// defaultMinStagingFree is the free space a staging filesystem must have for
// passthrough to be granted. Deliberately coarse: the file's eventual size is
// unknown at open, so this is a floor that keeps us away from a full staging
// area rather than a per-file reservation.
const defaultMinStagingFree = 256 << 20

// stagingFreeTTL caps how often we statfs the staging dir. Open is a hot path
// (a small-file loop can issue thousands per second); the free-space figure
// only needs to be approximately fresh.
const stagingFreeTTL = 200 * time.Millisecond

// waitInodeTimeout bounds how long an Open waits for a same-inode passthrough
// reconcile to finish before refusing the open outright (see waitInode).
const waitInodeTimeout = 30 * time.Second

// ptBacking is one registered kernel backing: a staging file plus the
// backing ID the kernel handed back for it. It outlives individual opens.
type ptBacking struct {
	path      string
	f         *os.File
	backingID int32
	// refs counts the write-capable opens using this backing: the original
	// passthrough writer plus any shared reopen with write access (see
	// share). The reconcile that lands the data runs when the last of them
	// releases; earlier releases only drop their reference. Guarded by
	// passthroughState.mu.
	refs int
	// mu serializes staging-content copies for this backing: fsync-time
	// copies against each other and against the final release-time
	// reconcile, from whichever open issues them.
	mu sync.Mutex
	// synced is the fingerprint of the staging content the last fsync-time
	// copy landed in JuiceFS, or nil if no fsync has copied yet. A later
	// copy (another fsync, or the release-time reconcile) is skipped when the
	// staging still matches it: the bytes are already in slices, and copying
	// them again would upload the whole file a second time. Every path that
	// changes the JuiceFS inode behind the staging's back must clear it (see
	// truncate). Guarded by mu.
	synced *ptSyncMark
}

type ptFile struct {
	ino Ino
	fh  uint64
	b   *ptBacking
	// writer is true for opens with write access: the original passthrough
	// writer and shared reopens that can write. Only these hold a busy
	// reference on the inode and a ref on the backing; a shared read-only
	// open just reads the staging file the kernel already has open.
	writer bool
	// shared is true for an open that joined an existing writer's backing
	// (see share) rather than checking out its own.
	shared bool
}

// ptSyncMark fingerprints staging content that has been copied into JuiceFS.
// Content-addressed, not stat-based: timestamps on the staging filesystem
// are too coarse to order a write against the copy that just finished, and
// mmap writes to a tmpfs backing do not update them at all.
type ptSyncMark struct {
	size uint64
	sum  [32]byte
}

// fingerprintStaging hashes the staging file at path from offset 0 through
// EOF, the same bytes copyStagingLocked would copy. Opened by path: reading
// through the registered backing fd can return stale or partial data.
func fingerprintStaging(path string) (*ptSyncMark, error) {
	rf, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer rf.Close()
	h := blake3.New()
	buf := utils.Alloc(4 << 20)
	defer utils.Free(buf)
	n, err := io.CopyBuffer(h, rf, buf)
	if err != nil {
		return nil, err
	}
	m := &ptSyncMark{size: uint64(n)}
	copy(m.sum[:], h.Sum(nil))
	return m, nil
}

// stagingUnchangedLocked reports whether the staging content still matches
// the fingerprint recorded by the last fsync-time copy. False when there is
// no mark, when the size differs (cheap, checked first), or when the content
// hash differs. Any error reads as "changed" so the caller falls back to the
// full copy, which is always safe. Caller holds pf.b.mu.
func (pf *ptFile) stagingUnchangedLocked() bool {
	if pf.b.synced == nil {
		return false
	}
	st, err := os.Stat(pf.b.path)
	if err != nil || uint64(st.Size()) != pf.b.synced.size {
		return false
	}
	m, err := fingerprintStaging(pf.b.path)
	if err != nil {
		return false
	}
	return m.size == pf.b.synced.size && m.sum == pf.b.synced.sum
}

func newPassthroughState(server *fuse.Server, dir string) *passthroughState {
	base := dir
	if base == "" {
		base = filepath.Join(os.TempDir(), "juicefs-passthrough")
	}
	_ = os.MkdirAll(base, 0700)
	// Isolate this mount's staging in a per-process subdir. Several juicefs
	// mounts (multiple volumes on a node, or a graceful-restart successor)
	// can share `base`; a flat shared dir would let one mount's pool-N.tmp
	// collide with another's, and — worse — a startup sweep of the shared dir
	// would unlink another live mount's in-use staging files, silently losing
	// the data of files whose close(2) already returned. A private subdir
	// removes both hazards, so we deliberately do NOT garbage-collect the
	// shared root here (a crashed predecessor's subdir is inert and can be
	// reaped out of band).
	sub, err := os.MkdirTemp(base, fmt.Sprintf("m%d-", os.Getpid()))
	if err != nil {
		logger.Warnf("passthrough: per-process staging dir under %s: %s; using base", base, err)
		sub = base
	}
	minFree := uint64(defaultMinStagingFree)
	if v := os.Getenv("JUICEFS_PASSTHROUGH_MIN_FREE"); v != "" {
		if n, err := strconv.ParseUint(v, 10, 64); err == nil {
			minFree = n
		} else {
			logger.Warnf("passthrough: ignoring JUICEFS_PASSTHROUGH_MIN_FREE=%q: %s", v, err)
		}
	}
	return &passthroughState{
		server:  server,
		dir:     sub,
		files:   make(map[uint64]*ptFile),
		busy:    make(map[Ino]int),
		minFree: minFree,
	}
}

// hasStagingHeadroom reports whether the staging filesystem has enough free
// space to be worth granting passthrough. Result is cached for stagingFreeTTL
// so a small-file loop does not statfs on every open. A statfs error is
// treated as "no headroom": we cannot confirm room, and the daemon path is
// always correct, just slower.
func (p *passthroughState) hasStagingHeadroom() bool {
	if p.minFree == 0 {
		return true
	}
	p.mu.Lock()
	if time.Since(p.freeCheckAt) < stagingFreeTTL {
		ok := p.freeOK
		p.mu.Unlock()
		return ok
	}
	p.mu.Unlock()

	var st syscall.Statfs_t
	ok := false
	var free uint64
	if err := syscall.Statfs(p.dir, &st); err != nil {
		logger.Warnf("passthrough: statfs staging %s: %s; treating as no headroom", p.dir, err)
	} else {
		free = st.Bavail * uint64(st.Bsize)
		ok = free >= p.minFree
	}

	p.mu.Lock()
	p.freeCheckAt = time.Now()
	p.freeOK = ok
	p.mu.Unlock()
	if !ok && free > 0 {
		p.lowSpaceWarn.Do(func() {
			logger.Warnf("passthrough: staging %s has %d bytes free, below the %d-byte floor; "+
				"falling back to the daemon write path (raise JUICEFS_PASSTHROUGH_MIN_FREE, "+
				"enlarge the staging volume, or point JUICEFS_PASSTHROUGH_DIR at a larger filesystem)",
				p.dir, free, p.minFree)
		})
	}
	return ok
}

// checkout registers a fresh backing for a new passthrough open. Always a new
// registration — see the passthroughState doc comment for why a backing is
// never reused across inodes.
func (p *passthroughState) checkout() (*ptBacking, bool) {
	p.mu.Lock()
	if p.disabled {
		p.mu.Unlock()
		return nil, false
	}
	p.seq++
	seq := p.seq
	p.mu.Unlock()

	path := filepath.Join(p.dir, fmt.Sprintf("pool-%d.tmp", seq))
	// O_EXCL: never open a staging file that already exists. The dir is
	// per-process and seq is monotonic, so a collision would signal a bug
	// (or a shared-dir fallback), and truncating a pre-existing file could
	// clobber another mount's in-use backing.
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		logger.Warnf("passthrough: open staging %s: %s", path, err)
		return nil, false
	}
	id, errno := p.server.RegisterBackingFd(&fuse.BackingMap{Fd: int32(f.Fd())})
	if errno != 0 {
		_ = f.Close()
		_ = os.Remove(path)
		// EPERM is permanent: the backing-registration ioctl needs
		// CAP_SYS_ADMIN in the init user namespace, and a process that
		// lacks it now will lack it for every open (e.g. a non-root
		// container whose added capabilities are bounding-set only).
		// Without this latch every write-open pays a doomed ioctl, a
		// staging create/remove, and a warning line (ENG26-869 saw 2000+
		// per run). Other errnos may be transient; keep trying those.
		if errno == syscall.EPERM {
			p.mu.Lock()
			p.disabled = true
			p.mu.Unlock()
			logger.Warnf("passthrough: RegisterBackingFd(%s): %s; disabling passthrough for this mount "+
				"(the ioctl needs CAP_SYS_ADMIN in the init user namespace — run the mount as root, "+
				"or use a mount broker that performs registrations node-side)", path, errno)
			return nil, false
		}
		logger.Warnf("passthrough: RegisterBackingFd(%s): %s", path, errno)
		return nil, false
	}
	return &ptBacking{path: path, f: f, backingID: id}, true
}

// retire tears a backing down for good: unregisters the kernel registration,
// closes the local fd, and removes the staging file. Never reused by another
// inode — see the passthroughState doc comment.
func (p *passthroughState) retire(b *ptBacking) {
	if errno := p.server.UnregisterBackingFd(b.backingID); errno != 0 {
		logger.Warnf("passthrough: UnregisterBackingFd(%d): %s", b.backingID, errno)
	}
	_ = b.f.Close()
	if err := os.Remove(b.path); err != nil && !os.IsNotExist(err) {
		// Data is already safely in JuiceFS at this point, so this is a
		// leaked-disk-space concern, not a durability one — but a silently
		// swallowed error here means nothing points an operator at the file.
		logger.Warnf("passthrough: remove staging %s: %s", b.path, err)
	}
}

func isWriteOpen(flags uint32) bool {
	acc := flags & uint32(syscall.O_ACCMODE)
	return acc == syscall.O_WRONLY || acc == syscall.O_RDWR
}

// tryOpen sets up passthrough for a write-opened file that is EMPTY at open.
// It returns the kernel backing ID and true on success; callers then set
// FOPEN_PASSTHROUGH and OpenOut.BackingID. On any failure it returns false and
// the caller falls back to the normal (daemon) path — passthrough is purely an
// optimization.
//
// emptyAtOpen MUST be true only when the file has no pre-existing content the
// open can observe or extend: a fresh Create, or an Open with O_TRUNC, or a
// zero-length file. This is a correctness gate, not a heuristic. The backing
// staging file always starts empty and FOPEN_PASSTHROUGH diverts reads, writes
// AND mmap to it, so enabling it on a non-empty file would (a) serve reads as
// zeros/EOF instead of real content (read-modify-write corruption), and (b)
// make reconcile — which copies staging linearly from offset 0 — overwrite the
// file's real prefix with holes for O_APPEND / sparse / seek-write patterns.
func (p *passthroughState) tryOpen(ino Ino, fh uint64, flags uint32, emptyAtOpen bool) (int32, bool) {
	if p == nil || !isWriteOpen(flags) || !emptyAtOpen {
		return 0, false
	}
	// Never hand JuiceFS's internal/control files (.control, .stats, .config,
	// ...) to the kernel via a backing file. Those inodes are served by the
	// daemon's in-process handlers; passthrough would silently divert their
	// I/O to a plain temp file, breaking the control protocol — e.g. the
	// `juicefs checkpoint` verb, whose write/read on .control would go to the
	// backing file and never reach the handler, hanging the command.
	if vfs.IsSpecialNode(ino) {
		return 0, false
	}
	// One passthrough writer per inode at a time. A second write-open while an
	// earlier one is still open — or while its reconcile is in flight (during
	// which the metadata size still reads 0, so emptyAtOpen looks true again) —
	// would get its own empty backing whose linear reconcile overwrites the
	// first's data. Reserve the inode; overlapping opens fall to the daemon
	// path, which serializes correctly. Checked before any server call so the
	// disabled latch and a drain-for-handover pause short-circuit cheaply.
	p.mu.Lock()
	if p.disabled || p.paused || p.busy[ino] > 0 {
		p.mu.Unlock()
		return 0, false
	}
	p.busy[ino]++
	p.mu.Unlock()
	release := func() {
		p.mu.Lock()
		p.releaseBusyLocked(ino)
		p.mu.Unlock()
	}

	// Check headroom after the busy reservation (so the cheap latches win) but
	// before registering a backing: granting passthrough onto a full staging
	// filesystem is strictly worse than not granting it.
	if !p.hasStagingHeadroom() {
		release()
		return 0, false
	}

	if !p.server.SupportsPassthrough() {
		p.warnOne.Do(func() {
			logger.Warnf("FUSE passthrough requested but not supported by the kernel; falling back")
		})
		release()
		return 0, false
	}
	b, ok := p.checkout()
	if !ok {
		release()
		return 0, false
	}
	p.mu.Lock()
	b.refs = 1
	p.files[fh] = &ptFile{ino: ino, fh: fh, b: b, writer: true}
	p.mu.Unlock()
	return b.backingID, true
}

// canShare reports whether ino currently has a live passthrough backing an
// open could join (see share). Cheap and lock-scoped; the answer can change
// before share runs, which is why share re-checks under the same lock.
func (p *passthroughState) canShare(ino Ino) bool {
	if p == nil {
		return false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.liveBackingLocked(ino) != nil
}

// liveBackingLocked returns the backing of any open on ino still in p.files,
// i.e. one whose reconcile has not started (reconcile removes the entry
// first). Caller holds p.mu.
func (p *passthroughState) liveBackingLocked(ino Ino) *ptBacking {
	for _, f := range p.files {
		if f.ino == ino {
			return f.b
		}
	}
	return nil
}

// share joins a new open of ino onto the backing of a passthrough writer
// that is still open, so the kernel serves the new fd from the same staging
// file and both see the same bytes. This is what a reopen-while-writing
// needs — git index-pack and uv both write a file and reopen it before
// closing the writer — and it is what made those fail: waitInode cannot
// end while the writer holds its fd, so every such reopen sat out the full
// timeout and got EAGAIN (2026-09-04 → 09-12, every git fetch into a volume).
//
// A read-only share holds nothing: the kernel keeps its own reference to the
// backing file, so reads keep working even after the writer's reconcile
// retires the registration and unlinks the path. A write-capable share
// counts as a writer — busy on the inode, a ref on the backing — so the
// reconcile waits for it (its writes would otherwise land in a staging file
// nobody copies again). Refused while draining for handover, when a new
// writer could outlive the drain.
func (p *passthroughState) share(ino Ino, fh uint64, flags uint32) (int32, bool) {
	if p == nil {
		return 0, false
	}
	writer := isWriteOpen(flags)
	p.mu.Lock()
	defer p.mu.Unlock()
	b := p.liveBackingLocked(ino)
	if b == nil || (writer && p.paused) {
		return 0, false
	}
	if writer {
		p.busy[ino]++
		b.refs++
	}
	p.files[fh] = &ptFile{ino: ino, fh: fh, b: b, writer: writer, shared: true}
	return b.backingID, true
}

// fixupLength raises entry's length to the live staging size while a
// passthrough backing is open for ino (see liveSize). Never lowers it.
func (p *passthroughState) fixupLength(ino Ino, entry *meta.Entry) {
	if p == nil || entry == nil || entry.Attr == nil {
		return
	}
	if sz, ok := p.liveSize(ino); ok && sz > entry.Attr.Length {
		entry.Attr.Length = sz
	}
}

// liveSize returns the current size of ino's staging file while a
// passthrough backing is live for it. Until reconcile lands, the metadata
// still says 0 (or the pre-write size), which is what a stat would report
// — wrong for anyone sizing a file they, or a sibling fd, just wrote.
func (p *passthroughState) liveSize(ino Ino) (uint64, bool) {
	if p == nil {
		return 0, false
	}
	p.mu.Lock()
	b := p.liveBackingLocked(ino)
	p.mu.Unlock()
	if b == nil {
		return 0, false
	}
	st, err := b.f.Stat()
	if err != nil {
		return 0, false
	}
	return uint64(st.Size()), true
}

// releaseBusyLocked drops one busy reference for ino. Caller holds p.mu.
func (p *passthroughState) releaseBusyLocked(ino Ino) {
	if n := p.busy[ino]; n <= 1 {
		delete(p.busy, ino)
	} else {
		p.busy[ino] = n - 1
	}
}

// waitInode blocks (bounded) until ino has no passthrough open or in-flight
// reconcile, returning true once it does. Called at the start of Open so a
// reopen of a file this session just wrote via passthrough observes the
// reconciled state: until reconcile lands, the metadata size still reads 0
// and a new write (passthrough OR daemon) would race the reconcile's linear
// copy and silently lose data. After a successful wait the file's size is
// authoritative, so tryOpen's emptyAtOpen gate correctly sends the reopen
// down the daemon path.
//
// On timeout it returns false. The metadata size may still be stale (an
// in-flight reconcile hasn't landed it), so the caller MUST NOT treat it as
// authoritative — proceeding as if the wait had succeeded would silently
// serve stale/empty content as if it were the real, reconciled file.
func (p *passthroughState) waitInode(ino Ino, timeout time.Duration) bool {
	if p == nil {
		return true
	}
	deadline := time.Now().Add(timeout)
	for {
		p.mu.Lock()
		busy := p.busy[ino] > 0
		p.mu.Unlock()
		if !busy {
			return true
		}
		if time.Now().After(deadline) {
			logger.Warnf("passthrough: waited %s for in-flight reconcile of ino %d; refusing this open rather than risk stale content", timeout, ino)
			return false
		}
		time.Sleep(2 * time.Millisecond)
	}
}

// reconcile flushes a passthrough staging file back into JuiceFS slices, then
// tears down the backing registration. Called on release, before vfs.Release.
func (p *passthroughState) reconcile(ctx vfs.Context, v *vfs.VFS, fh uint64) {
	if p == nil {
		return
	}
	p.mu.Lock()
	pf := p.files[fh]
	if pf != nil {
		delete(p.files, fh)
	}
	last := false
	if pf != nil && pf.writer {
		pf.b.refs--
		last = pf.b.refs == 0
	}
	p.mu.Unlock()
	if pf == nil || !pf.writer {
		// A shared read-only open: it borrowed the writer's backing and owns
		// no data; the kernel drops its own reference to the staging file.
		return
	}
	if !last {
		// Another write-capable open still uses this backing; whichever
		// releases last lands the data. Drop this open's busy reference.
		p.mu.Lock()
		p.releaseBusyLocked(pf.ino)
		p.mu.Unlock()
		return
	}
	// Free the inode for a new passthrough open only once its data has fully
	// landed (or been preserved on failure) — held across the whole reconcile.
	defer func() {
		p.mu.Lock()
		p.releaseBusyLocked(pf.ino)
		p.mu.Unlock()
	}()
	// Fence for consistency points now that we know this is a passthrough
	// release with data to land: the application's close(2) has already
	// returned, but the data still lives only in the staging file until the
	// copy below finishes. A checkpoint or commit that ran concurrently would
	// flush+snapshot without these writes and publish a mid-copy (short) file.
	// The external-flush counter lets those paths wait for in-flight
	// reconciles first. Scoped past the pf==nil check so plain (non-
	// passthrough) releases don't add spurious fence contention.
	v.BeginExternalFlush()
	defer v.EndExternalFlush()
	// The kernel stops issuing passthrough I/O for this open once its release
	// is processed, and reconcile runs from the RELEASE handler — after the
	// application's last close, so no writes are in flight. The registration
	// is retired for good once reconciled (see retire) rather than reused by
	// a future open, which may belong to a different inode; read the staging
	// content through a fresh path-open fd for a clean sequential pass
	// (reading via the registered backing fd can return stale/partial data).
	b := pf.b
	done := false
	defer func() {
		if !done {
			// Reconcile failed: the staging file is the ONLY copy of data whose
			// close(2) already returned 0, so do NOT delete it — preserve it as
			// an .orphan sibling for manual recovery and log loudly. Drop the
			// kernel registration and the live fd so the backing isn't reused
			// with stale data.
			if errno := p.server.UnregisterBackingFd(b.backingID); errno != 0 {
				logger.Warnf("passthrough: UnregisterBackingFd(%d): %s", b.backingID, errno)
			}
			_ = b.f.Close()
			orphan := b.path + fmt.Sprintf(".orphan-%d", pf.ino)
			if err := os.Rename(b.path, orphan); err != nil {
				logger.Errorf("passthrough: reconcile of ino %d FAILED and staging %s could not be preserved: %s", pf.ino, b.path, err)
			} else {
				logger.Errorf("passthrough: reconcile of ino %d FAILED; unreconciled data preserved at %s", pf.ino, orphan)
			}
		}
	}()

	pf.b.mu.Lock()
	defer pf.b.mu.Unlock()
	var off uint64
	if pf.stagingUnchangedLocked() {
		// An fsync already copied exactly this content into slices; a second
		// full copy would re-upload the whole file (measured: every
		// write-fsync-close paid 2x its size in object PUTs). Skip the copy;
		// the flush below still lands anything the writer holds.
		off = pf.b.synced.size
		logger.Debugf("passthrough: ino %d staging unchanged since fsync, not recopying %d bytes", pf.ino, off)
	} else {
		var ok bool
		off, _, ok = p.copyStagingLocked(ctx, v, pf)
		if !ok {
			return
		}
	}
	if e := v.Flush(ctx, pf.ino, fh, 0); e != 0 {
		logger.Errorf("passthrough: reconcile flush ino %d: %s", pf.ino, e)
		return
	}
	// Passthrough writes bypassed the daemon, so the kernel's cached size and
	// page data for this inode are stale (size is still 0 from the empty
	// create). Now that the slices + metadata are committed, invalidate both so
	// readers in this mount session see the reconciled file (read-your-writes).
	p.server.InodeNotify(uint64(pf.ino), -1, 0)         // attributes (size/mtime)
	p.server.InodeNotify(uint64(pf.ino), 0, int64(off)) // data range
	// Data is safely in JuiceFS: retire the registration and staging file for
	// good. No truncate needed — retire removes the file outright.
	done = true
	p.retire(b)
}

// copyStagingLocked copies the full staging content of pf into JuiceFS slices
// via the normal writer path. Returns the byte count and false on any read or
// write error — a read error mid-copy must NOT be mistaken for end-of-file,
// or the caller would flush+commit a truncated file as if complete. Always a
// FULL copy from offset 0: passthrough writes land in the backing at
// arbitrary offsets, so an incremental "since last copy" scheme would miss
// overwrites of already-copied ranges. Also returns the blake3 fingerprint of
// the bytes copied, so an fsync-time copy can record what is now in slices
// and a later copy can be skipped if the staging has not changed since.
// Caller holds pf.b.mu.
func (p *passthroughState) copyStagingLocked(ctx vfs.Context, v *vfs.VFS, pf *ptFile) (uint64, [32]byte, bool) {
	var sum [32]byte
	rf, err := os.Open(pf.b.path)
	if err != nil {
		logger.Errorf("passthrough: reopen staging %s: %s", pf.b.path, err)
		return 0, sum, false
	}
	defer rf.Close()
	h := blake3.New()
	// Pooled/accounted allocation: vfs.writer's and vfs/compact's memory
	// backpressure (utils.AllocMemory()-store.UsedMemory()) only sees bytes
	// allocated through utils.Alloc. A raw make() here would be invisible to
	// that accounting, so concurrent reconciles could balloon RSS well past
	// the configured buffer budget without the compactor/writer ever
	// throttling in response.
	buf := utils.Alloc(4 << 20)
	defer utils.Free(buf)
	var off uint64
	for {
		n, err := rf.Read(buf)
		if n > 0 {
			// vfs.Write may retain the buffer until flush; give each chunk its
			// own backing array so the next Read doesn't corrupt a pending slice.
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			_, _ = h.Write(chunk)
			if e := v.Write(ctx, pf.ino, chunk, off, pf.fh); e != 0 {
				logger.Errorf("passthrough: staging copy write ino %d off %d: %s", pf.ino, off, e)
				return off, sum, false
			}
			off += uint64(n)
		}
		if err != nil {
			if err != io.EOF {
				logger.Errorf("passthrough: read staging %s at off %d: %s", pf.b.path, off, err)
				return off, sum, false
			}
			break
		}
	}
	copy(sum[:], h.Sum(nil))
	return off, sum, true
}

// fsync makes fsync(2)/fdatasync(2) honest for a passthrough open. The
// kernel diverts only read/write/mmap to the backing file — FSYNC still
// reaches the daemon, whose writer has no data for this fh, so plain
// vfs.Fsync would report success while every byte still sits in a local
// staging file that a crash before release would lose. Instead, copy the
// staging content into JuiceFS slices now (the open stays live and the
// backing stays registered) and then flush the writer, giving the caller
// exactly the durability a non-passthrough fsync provides. The copy records
// a fingerprint of what it landed; the release-time reconcile — still the
// authority on final content — recopies only if the staging changed after
// this point, so a write-fsync-close sequence uploads its bytes once, not
// twice. A second fsync with nothing new likewise copies nothing.
//
// Returns handled=false when fh is not a passthrough open (caller proceeds
// with the normal path).
func (p *passthroughState) fsync(ctx vfs.Context, v *vfs.VFS, fh uint64) (bool, syscall.Errno) {
	if p == nil {
		return false, 0
	}
	p.mu.Lock()
	pf := p.files[fh]
	p.mu.Unlock()
	if pf == nil {
		return false, 0
	}
	pf.b.mu.Lock()
	defer pf.b.mu.Unlock()
	// Fence: a checkpoint/commit racing this copy must wait for it, exactly
	// as for a release-time reconcile, or it could snapshot a mid-copy state
	// of a file the application believes it just made durable.
	v.BeginExternalFlush()
	defer v.EndExternalFlush()
	if !pf.stagingUnchangedLocked() {
		off, sum, ok := p.copyStagingLocked(ctx, v, pf)
		if !ok {
			// The open is still live and release will retry the copy; report
			// the failure so the application does not trust this fsync.
			return true, syscall.EIO
		}
		pf.b.synced = &ptSyncMark{size: off, sum: sum}
	}
	if e := v.Fsync(ctx, pf.ino, 0, fh); e != 0 {
		return true, e
	}
	return true, 0
}

// truncate mirrors a successful SETATTR size change onto the inode's live
// passthrough backing, if any. truncate(2)/ftruncate(2) reach the daemon —
// the kernel never diverts SETATTR to the backing — so without this the
// backing keeps its old length and diverges from the size the caller just
// set: reads through the passthrough fd see the old data/EOF, and the
// release-time reconcile (linear copy of the staging, authority on final
// content) would silently undo the truncate.
func (p *passthroughState) truncate(ino Ino, size uint64) {
	if p == nil {
		return
	}
	p.mu.Lock()
	var pf *ptFile
	for _, f := range p.files {
		if f.ino == ino {
			pf = f
			break
		}
	}
	p.mu.Unlock()
	if pf == nil {
		return
	}
	pf.b.mu.Lock()
	defer pf.b.mu.Unlock()
	// The JuiceFS inode just changed behind the staging's back: a later
	// staging that happens to match the fsync-time fingerprint again (shrink,
	// then rewrite the same tail) would otherwise skip a copy the truncated
	// inode needs. Forget the mark so the next copy is a full one.
	pf.b.synced = nil
	if err := pf.b.f.Truncate(int64(size)); err != nil {
		logger.Errorf("passthrough: mirror truncate ino %d to %d on %s: %s", ino, size, pf.b.path, err)
	}
}

// drain blocks new passthrough opens and waits (bounded) until no inode has
// a live passthrough open or in-flight reconcile. Used before a smooth
// upgrade (SIGHUP handover): a passthrough open's data exists only in this
// process's staging files, which the successor has no record of, so handing
// over while any are live silently loses every byte written through them.
// On timeout it re-enables passthrough and returns false — the caller must
// refuse the handover and keep serving.
func (p *passthroughState) drain(timeout time.Duration) bool {
	p.mu.Lock()
	p.paused = true
	p.mu.Unlock()
	deadline := time.Now().Add(timeout)
	for {
		p.mu.Lock()
		n := len(p.busy)
		p.mu.Unlock()
		if n == 0 {
			return true
		}
		if time.Now().After(deadline) {
			p.mu.Lock()
			p.paused = false
			p.mu.Unlock()
			logger.Warnf("passthrough: %d inode(s) still have live passthrough opens or reconciles after %s", n, timeout)
			return false
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// reconcileAll proactively reconciles every currently-tracked passthrough
// open. reconcile() normally only runs from the RELEASE handler as each
// open closes, but a force or lazy unmount can tear down the FUSE
// connection without ever delivering RELEASE for opens that were still
// live — some of which may have data whose close(2) already returned
// successfully to the application. Without this, that data would simply be
// abandoned in a staging file nothing ever looks at again. Call this from
// the shutdown path BEFORE the connection is torn down, while backing
// registrations and InodeNotify are still valid.
func (p *passthroughState) reconcileAll(ctx vfs.Context, v *vfs.VFS) {
	if p == nil {
		return
	}
	p.mu.Lock()
	fhs := make([]uint64, 0, len(p.files))
	for fh := range p.files {
		fhs = append(fhs, fh)
	}
	p.mu.Unlock()
	for _, fh := range fhs {
		p.reconcile(ctx, v, fh)
	}
}
