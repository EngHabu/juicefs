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
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// A passthrough writer with its staging file, as tryOpen would leave it,
// without a kernel: enough to exercise the share/refs/busy bookkeeping.
func fakeWriter(t *testing.T, p *passthroughState, ino Ino, fh uint64, content []byte) *ptBacking {
	t.Helper()
	path := filepath.Join(p.dir, "pool-1.tmp")
	if err := os.MkdirAll(p.dir, 0o700); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(content); err != nil {
		t.Fatal(err)
	}
	b := &ptBacking{path: path, f: f, backingID: 7, refs: 1}
	p.mu.Lock()
	p.busy[ino]++
	p.files[fh] = &ptFile{ino: ino, fh: fh, b: b, writer: true}
	p.mu.Unlock()
	return b
}

// TestShareReopenWhileWriterOpen: the case every git fetch into a volume hit
// — a file written through passthrough is reopened before its writer closes.
// The reopen must join the writer's backing immediately instead of waiting
// on the inode (which cannot clear while the writer holds its fd).
func TestShareReopenWhileWriterOpen(t *testing.T) {
	p := &passthroughState{dir: filepath.Join(t.TempDir(), "pt"), files: make(map[uint64]*ptFile), busy: make(map[Ino]int)}
	const ino = Ino(42)
	b := fakeWriter(t, p, ino, 1, []byte("sixty-one kilobytes, notionally"))

	if !p.canShare(ino) {
		t.Fatal("canShare must be true while the writer is open")
	}
	if p.canShare(Ino(43)) {
		t.Fatal("canShare must be false for an inode with no backing")
	}
	// Read-only reopen: same backing, no busy or ref accounting.
	id, ok := p.share(ino, 2, uint32(syscall.O_RDONLY))
	if !ok || id != 7 {
		t.Fatalf("share(reader) = %d, %v; want 7, true", id, ok)
	}
	if p.busy[ino] != 1 || b.refs != 1 {
		t.Fatalf("a read-only share must not take busy/refs: busy=%d refs=%d", p.busy[ino], b.refs)
	}
	// Write-capable reopen: counts as a writer.
	if _, ok := p.share(ino, 3, uint32(syscall.O_RDWR)); !ok {
		t.Fatal("share(writer) refused")
	}
	if p.busy[ino] != 2 || b.refs != 2 {
		t.Fatalf("a write share must take busy and a ref: busy=%d refs=%d", p.busy[ino], b.refs)
	}
	// The staging size is what stat must report meanwhile.
	if sz, ok := p.liveSize(ino); !ok || sz != uint64(len("sixty-one kilobytes, notionally")) {
		t.Fatalf("liveSize = %d, %v", sz, ok)
	}

	// Releasing the reader: nothing to reconcile, nothing to free.
	p.reconcile(nil, nil, 2)
	if _, still := p.files[2]; still {
		t.Fatal("reader entry not removed")
	}
	if p.busy[ino] != 2 || b.refs != 2 {
		t.Fatalf("reader release changed accounting: busy=%d refs=%d", p.busy[ino], b.refs)
	}
	// Releasing one of two writers: drops its ref and busy, keeps the backing
	// (the other writer may still be writing into it).
	p.reconcile(nil, nil, 3)
	if p.busy[ino] != 1 || b.refs != 1 {
		t.Fatalf("first writer release: busy=%d refs=%d, want 1/1", p.busy[ino], b.refs)
	}
	if _, err := os.Stat(b.path); err != nil {
		t.Fatalf("backing retired while a writer still uses it: %v", err)
	}
	if !p.canShare(ino) {
		t.Fatal("the remaining writer must still be shareable")
	}
	// A new open while the writer is open never waits: waitInode is only
	// for the reconcile-in-flight case, and share is what Open uses first.
	done := make(chan bool, 1)
	go func() { done <- p.waitInode(ino, 50*time.Millisecond) }()
	if <-done {
		t.Fatal("waitInode returned true while a writer holds the inode")
	}
}

// TestShareRefusedWithoutWriter: no live backing, no share — Open falls back
// to the reconcile wait / daemon path exactly as before.
func TestShareRefusedWithoutWriter(t *testing.T) {
	p := &passthroughState{dir: t.TempDir(), files: make(map[uint64]*ptFile), busy: make(map[Ino]int)}
	if _, ok := p.share(Ino(1), 9, uint32(syscall.O_RDONLY)); ok {
		t.Fatal("share must refuse an inode with no open writer")
	}
	if _, ok := p.liveSize(Ino(1)); ok {
		t.Fatal("liveSize must report nothing without a backing")
	}
	// Draining for handover: read shares are fine, new writers are not.
	fakeWriter(t, p, Ino(2), 1, []byte("x"))
	p.paused = true
	if _, ok := p.share(Ino(2), 2, uint32(syscall.O_RDONLY)); !ok {
		t.Fatal("read share must work while paused")
	}
	if _, ok := p.share(Ino(2), 3, uint32(syscall.O_WRONLY)); ok {
		t.Fatal("write share must be refused while paused")
	}
}
