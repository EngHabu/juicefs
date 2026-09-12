//go:build linux

/*
 * JuiceFS, Copyright 2026 Juicedata, Inc.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

package fuse

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func writeStaging(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

// A file that has never been fsync-copied carries no mark and must always be
// copied in full: the skip is an optimization gated on a recorded fingerprint,
// never on the absence of one.
func TestStagingUnchangedRequiresMark(t *testing.T) {
	path := filepath.Join(t.TempDir(), "staging")
	writeStaging(t, path, bytes.Repeat([]byte("a"), 1<<20))
	pf := &ptFile{b: &ptBacking{path: path}}
	if pf.stagingUnchangedLocked() {
		t.Fatalf("no sync mark must mean 'changed' (full copy)")
	}
}

// The mark must match the content it was taken from and only that content:
// an appended byte, a same-size rewrite, and a missing file all read as
// changed; restoring the original bytes reads as unchanged again.
func TestStagingUnchangedTracksContent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "staging")
	orig := bytes.Repeat([]byte("0123456789abcdef"), 1<<18) // 4 MiB, crosses the copy buffer
	writeStaging(t, path, orig)
	m, err := fingerprintStaging(path)
	if err != nil {
		t.Fatal(err)
	}
	if m.size != uint64(len(orig)) {
		t.Fatalf("fingerprint size %d, want %d", m.size, len(orig))
	}
	pf := &ptFile{b: &ptBacking{path: path, synced: m}}
	if !pf.stagingUnchangedLocked() {
		t.Fatalf("identical content must read as unchanged")
	}

	writeStaging(t, path, append(append([]byte{}, orig...), 'x'))
	if pf.stagingUnchangedLocked() {
		t.Fatalf("an appended byte must read as changed")
	}

	flipped := append([]byte{}, orig...)
	flipped[len(flipped)/2] ^= 0xff
	writeStaging(t, path, flipped)
	if pf.stagingUnchangedLocked() {
		t.Fatalf("a same-size rewrite must read as changed")
	}

	writeStaging(t, path, orig)
	if !pf.stagingUnchangedLocked() {
		t.Fatalf("restored content must read as unchanged")
	}

	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if pf.stagingUnchangedLocked() {
		t.Fatalf("an unreadable staging must read as changed (fall back to the full copy)")
	}
}

// An empty staging fingerprints too: a zero-length file fsync'd then closed
// must not trigger a copy either way, and must not be mistaken for "no mark".
func TestStagingUnchangedEmptyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "staging")
	writeStaging(t, path, nil)
	m, err := fingerprintStaging(path)
	if err != nil {
		t.Fatal(err)
	}
	pf := &ptFile{b: &ptBacking{path: path, synced: m}}
	if !pf.stagingUnchangedLocked() {
		t.Fatalf("empty file with its own mark must read as unchanged")
	}
}

// truncate reaches the daemon (the kernel never diverts SETATTR), so the
// JuiceFS inode changes behind the staging's back. The mark must be dropped:
// otherwise shrink-then-rewrite-the-same-tail leaves the staging matching the
// fsync-time fingerprint and the release would skip the copy the truncated
// inode needs. Regression guard for a silent data-loss path.
func TestTruncateClearsSyncMark(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "staging")
	writeStaging(t, path, bytes.Repeat([]byte("z"), 4096))
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	m, err := fingerprintStaging(path)
	if err != nil {
		t.Fatal(err)
	}
	pf := &ptFile{ino: Ino(9), fh: 3, b: &ptBacking{path: path, f: f, synced: m}, writer: true}
	p := &passthroughState{dir: dir, files: map[uint64]*ptFile{3: pf}, busy: map[Ino]int{Ino(9): 1}}
	p.truncate(Ino(9), 1024)
	if pf.b.synced != nil {
		t.Fatalf("truncate must clear the sync mark")
	}
	if st, err := os.Stat(path); err != nil || st.Size() != 1024 {
		t.Fatalf("truncate must still mirror the size onto the backing: %v %v", st, err)
	}
}
