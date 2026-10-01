#!/usr/bin/env python3
"""Leaf-PC wall-clock sampler via ptrace — no perf/PMU needed.

Works under perf_event_paranoid=4 + yama ptrace_scope=1: the sampler forks
the target itself, so every tracee is our own child (ancestors may ptrace).

Spawns CMD once, samples all its threads every TICK_MS for DURATION_S,
ignoring the first SKIP_MS of each process (skip parse/init). Resolves RIPs
via /proc/maps + readelf symbol tables (size-bounded when known, else
nearest-symbol). `setarch -R` keeps PIE bases stable across runs.

Usage: ptrace_sample.py <duration_s> <tick_ms> <skip_ms> <out.txt> <cmd...>
Out: one `count fn(module)` per line, sorted desc.
"""
import bisect
import ctypes
import os
import subprocess
import sys
import time
from collections import Counter

libc = ctypes.CDLL("libc.so.6", use_errno=True)
PTRACE_ATTACH = 16
PTRACE_DETACH = 17
PTRACE_GETREGS = 12

REG_NAMES = (
    "r15 r14 r13 r12 rbp rbx r11 r10 r9 r8 rax rcx rdx rsi rdi "
    "orig_rax rip cs eflags rsp ss fs_base gs_base ds es fs gs"
).split()


class user_regs_struct(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in REG_NAMES]


def get_rip(tid: int):
    if libc.ptrace(PTRACE_ATTACH, tid, None, None) != 0:
        return None
    try:
        try:
            os.waitpid(tid, 0)
        except ChildProcessError:
            pass
        regs = user_regs_struct()
        if libc.ptrace(PTRACE_GETREGS, tid, None, ctypes.byref(regs)) != 0:
            return None
        return regs.rip
    finally:
        libc.ptrace(PTRACE_DETACH, tid, None, None)


def read_maps(pid: int):
    maps = []
    try:
        with open(f"/proc/{pid}/maps") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 6 or "x" not in parts[1]:
                    continue
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
                maps.append((lo, hi, int(parts[2], 16), parts[5]))
    except OSError:
        pass
    return maps


class ElfFile:
    """Symtab + LOAD phdrs for one ELF. Symbol lookup takes a *file offset*
    (rip - map_start + map_offset) — LOAD segments can have p_vaddr != p_offset,
    so translating through the phdr is required (the 0x1000 p_offset/p_vaddr
    skew in rustc output will otherwise attribute every PC to the function
    ~4KB earlier in .text)."""

    def __init__(self, path: str):
        import subprocess as sp

        self.syms = []
        self.loads = []  # (p_offset, filesz, p_vaddr)
        try:
            out = sp.run(
                ["readelf", "-sW", path],
                capture_output=True, text=True, timeout=30,
            ).stdout
            for ln in out.splitlines():
                parts = ln.split(None, 7)
                # Num: Value Size Type Bind Vis Ndx Name
                if (
                    len(parts) == 8
                    and parts[0].rstrip(":").isdigit()
                    and parts[3] == "FUNC"
                    and parts[6] not in ("UND", "ABS")
                ):
                    self.syms.append(
                        (int(parts[1], 16), int(parts[2]), parts[7].split("(")[0])
                    )
            ph = sp.run(
                ["readelf", "-lW", path],
                capture_output=True, text=True, timeout=30,
            ).stdout
            for ln in ph.splitlines():
                parts = ln.split()
                if parts and parts[0] == "LOAD":
                    # LOAD offset vaddr paddr filesz memsz flags align
                    self.loads.append(
                        (int(parts[1], 16), int(parts[4], 16), int(parts[2], 16))
                    )
        except Exception:
            pass
        self.syms.sort()
        self.addrs = [a for a, _, _ in self.syms]
        self.base = os.path.basename(path)

    def lookup(self, foff: int):
        # file offset -> vaddr through the containing LOAD segment
        vaddr = None
        for p_off, filesz, p_vaddr in self.loads:
            if p_off <= foff < p_off + filesz:
                vaddr = foff - p_off + p_vaddr
                break
        if vaddr is None:
            vaddr = foff  # no phdr info: assume congruent layout
        i = bisect.bisect_right(self.addrs, vaddr) - 1
        if i < 0:
            return None
        a, size, n = self.syms[i]
        if size and a <= vaddr < a + size:
            return n
        if not size and vaddr - a < 0x10000:
            return n
        return None


def main() -> None:
    dur, tick_ms, skip_ms = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    out_path, cmd = sys.argv[4], sys.argv[5:]

    hist: Counter[str] = Counter()
    total = 0
    elfs: dict[str, ElfFile] = {}
    t_end = time.time() + dur

    proc = subprocess.Popen(
        ["setarch", "-R", *cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    t0 = time.time()
    while proc.poll() is None and time.time() < t_end:
        if (time.time() - t0) * 1000 < skip_ms:
            time.sleep(0.05)
            continue
        maps = read_maps(proc.pid)
        try:
            tids = os.listdir(f"/proc/{proc.pid}/task")
        except OSError:
            break
        for tid_s in tids:
            rip = get_rip(int(tid_s))
            if rip is None:
                continue
            total += 1
            name = None
            for lo, hi, off, path in maps:
                if lo <= rip < hi and path.startswith("/"):
                    st = elfs.get(path)
                    if st is None:
                        st = elfs[path] = ElfFile(path)
                    name = st.lookup(rip - lo + off)
                    if name is None:
                        name = f"{os.path.basename(path)}+0x{rip - lo + off:x}"
                    break
            if name is None:
                name = f"anon+0x{rip:x}"
            hist[name] += 1
        time.sleep(tick_ms / 1000.0)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(1)
        except subprocess.TimeoutExpired:
            proc.kill()
    with open(out_path, "w") as f:
        f.write(f"# samples={total}\n")
        for name, c in hist.most_common():
            f.write(f"{c} {name}\n")
    print(f"{total} samples -> {out_path}")


if __name__ == "__main__":
    main()
