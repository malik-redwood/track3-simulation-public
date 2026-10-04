#!/usr/bin/env python
"""Pre-flight checks on the submission binary, without needing Docker.

The dev box has no container runtime, so the image cannot be built or run here. These checks
verify the properties the image actually depends on, by parsing the ELF directly:

  1. it is a 64-bit x86-64 ELF        -- linux/amd64 as required
  2. it is statically linked          -- no PT_INTERP, so it runs in FROM scratch with no libc
  3. it has no shell dependency       -- no "/bin/sh" string, proving system()/popen() are gone
  4. it declares no shared libraries  -- nothing to resolve at run time

Check 3 is the one that caught a real bug: the first CLI shelled out for mkdir and directory
listing, which would have failed inside a scratch image that has no shell at all.

    python preflight.py simulate_linux
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

PT_INTERP = 3
PT_DYNAMIC = 2


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "simulate_linux")
    if not path.exists():
        print(f"MISSING: {path}", file=sys.stderr)
        return 2
    blob = path.read_bytes()
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    print(f"{path.name}: {len(blob):,} bytes\n")

    # ---- ELF header -------------------------------------------------
    check("ELF magic", blob[:4] == b"\x7fELF")
    if blob[:4] != b"\x7fELF":
        return 1
    ei_class = blob[4]
    e_type, e_machine = struct.unpack_from("<HH", blob, 16)
    check("64-bit", ei_class == 2, f"ei_class={ei_class}")
    check("x86-64", e_machine == 0x3E, f"e_machine=0x{e_machine:x}")
    check("executable (ET_EXEC=2, not a PIE/shared ET_DYN=3)", e_type == 2, f"e_type={e_type}")

    # ---- program headers --------------------------------------------
    e_phoff, = struct.unpack_from("<Q", blob, 32)
    e_phentsize, e_phnum = struct.unpack_from("<HH", blob, 54)
    ptypes = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, = struct.unpack_from("<I", blob, off)
        ptypes.append(p_type)
    check("statically linked (no PT_INTERP)", PT_INTERP not in ptypes,
          f"{e_phnum} program headers")
    check("no dynamic section (no PT_DYNAMIC)", PT_DYNAMIC not in ptypes)

    # ---- shell / dynamic-library strings ----------------------------
    for needle in (b"/bin/sh", b"/bin/bash"):
        check(f"no {needle.decode()} reference", needle not in blob,
              "system()/popen() would need one" if needle == b"/bin/sh" else "")
    for needle in (b"libc.so", b"ld-linux", b"libstdc++.so", b"libm.so"):
        check(f"no {needle.decode()} dependency", needle not in blob)

    # ---- the verbs are actually present -----------------------------
    for needle in (b"simulate-batch", b"--config", b"--batch-dir", b"trace.parquet",
                   b"message_trace.parquet", b"batch_events.json", b"PAR1"):
        check(f"contains {needle.decode()!r}", needle in blob)

    print()
    print("PRE-FLIGHT OK -- safe to build the image" if ok
          else "PRE-FLIGHT FAILED -- fix before building")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
