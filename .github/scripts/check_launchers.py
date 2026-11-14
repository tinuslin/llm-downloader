"""CI guard: keep the one-click launchers byte-exact across platforms.

cmd.exe requires CRLF + pure ASCII + no BOM for start.bat; POSIX shells
require LF for start.sh. A stray conversion silently breaks double-click.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    bat = (ROOT / "start.bat").read_bytes()
    if bat[:3] == b"\xef\xbb\xbf":
        fail("start.bat has a UTF-8 BOM")
    if not all(c < 128 for c in bat):
        fail("start.bat contains non-ASCII bytes")
    if bat.replace(b"\r\n", b"").count(b"\n"):
        fail("start.bat has LF-only line endings")
    if bat.count(b"\r\n") < 10:
        fail("start.bat looks truncated")
    if bat.split(b"\r\n")[0].strip() != b"@echo off":
        fail("start.bat first line is not '@echo off'")

    sh = (ROOT / "start.sh").read_bytes()
    if sh[:3] == b"\xef\xbb\xbf":
        fail("start.sh has a UTF-8 BOM")
    if b"\r\n" in sh:
        fail("start.sh has CRLF line endings")

    crlf = bat.count(b"\r\n")
    lf = sh.count(b"\n")
    print(f"OK: start.bat {crlf} CRLF lines, start.sh {lf} LF lines")


if __name__ == "__main__":
    main()
