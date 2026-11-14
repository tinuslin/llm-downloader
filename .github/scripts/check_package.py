"""CI guard: Web UI must be present in both wheel and sdist."""

from pathlib import Path
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
NEEDLE = "llm_downloader/static/index.html"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    wheels = sorted(DIST.glob("*.whl"))
    sdists = sorted(DIST.glob("*.tar.gz"))
    if not wheels:
        fail("no wheel produced")
    if not sdists:
        fail("no sdist produced")

    with zipfile.ZipFile(wheels[-1]) as z:
        names = z.namelist()
    if NEEDLE not in names:
        fail(f"{wheels[-1].name} is missing {NEEDLE}")
    print(f"OK wheel : {wheels[-1].name} ({len(names)} entries)")

    with tarfile.open(sdists[-1], "r:gz") as t:
        names = t.getnames()
    if not any(n.endswith(NEEDLE) for n in names):
        fail(f"{sdists[-1].name} is missing {NEEDLE}")
    print(f"OK sdist: {sdists[-1].name} ({len(names)} entries)")


if __name__ == "__main__":
    main()
