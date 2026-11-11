"""Find the NISE checkout after this folder left the repository tree."""

from pathlib import Path


def nise_root() -> Path:
    candidates = (
        Path(r"c:\Users\EDY\Documents\GitHub\NISE"),
        Path("/root/autodl-tmp/NISE"),
    )
    for path in candidates:
        if (path / "run_nise_boltz2x.py").is_file():
            return path
    joined = " 或 ".join(str(path) for path in candidates)
    raise SystemExit(f"找不到 NISE 仓库：{joined}")
