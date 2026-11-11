"""Rewrite the three Boltz parents into the RFD3 ligand layout.

Boltz writes oxaliplatin as residue OXP on chain B, with the metal named PT1
and twelve hydrogens. Foundry partial diffusion reads the protein from the
input and appends the ligand named by `ligand: l:g`. The de novo campaign
uses that name on chain C, residue 1, metal PT, heavy atoms only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from _nise_root import nise_root
NISE = nise_root()
CAMPAIGN = NISE / "campaigns" / "oxaliplatin"
if str(NISE) not in sys.path:
    sys.path.insert(0, str(NISE))

from campaigns.oxaliplatin.ligand.build_oxaliplatin import BONDS, ORDER, _pdb_line  # noqa: E402
from campaigns.oxaliplatin.partial_diffusion_maturation import PARENTS  # noqa: E402

RUN_ROOT = CAMPAIGN / "nise_run_pocket"
OXALATE = ("C1", "C2", "O1", "O2", "O3", "O4")


def _atom_name(line: str) -> str:
    return line[12:16].strip()


def _element(line: str) -> str:
    if len(line) >= 78 and line[76:78].strip():
        return line[76:78].strip().upper()
    name = _atom_name(line)
    return "".join(ch for ch in name if ch.isalpha()).upper()[:2]


def _coord(line: str) -> np.ndarray:
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)


def ligand_heavy_coords(lines: list[str]) -> dict[str, np.ndarray]:
    coords: dict[str, np.ndarray] = {}
    for line in lines:
        if not line.startswith("HETATM"):
            continue
        name = _atom_name(line)
        if name == "PT1":
            name = "PT"
        element = _element(line)
        if element == "H" or name.startswith("H"):
            continue
        coords[name] = _coord(line)
    missing = [name for name, _element in ORDER if name not in coords]
    if missing:
        raise RuntimeError(f"ligand is missing {missing}")
    return coords


def oxalate_neighbors(protein: np.ndarray, ligand: dict[str, np.ndarray]) -> dict[str, int]:
    counts = {}
    for name in OXALATE:
        dist = np.linalg.norm(protein - ligand[name], axis=1)
        counts[name] = int(np.sum(dist < 5.0))
    return counts


def convert_one(src: Path, dest: Path) -> dict[str, int]:
    text = src.read_text(encoding="utf-8", errors="replace").splitlines()
    protein_lines = []
    protein_xyz = []
    for line in text:
        if not line.startswith("ATOM"):
            continue
        if _element(line) == "H":
            continue
        protein_lines.append(line)
        protein_xyz.append(_coord(line))
    if not protein_lines:
        raise RuntimeError(f"no protein atoms in {src}")
    ligand = ligand_heavy_coords(text)
    neighbors = oxalate_neighbors(np.vstack(protein_xyz), ligand)

    out: list[str] = []
    serial = 1
    for line in protein_lines:
        out.append(f"ATOM  {serial:5d}" + line[11:])
        serial += 1
    out.append("TER")
    ligand_serial = {}
    for name, element in ORDER:
        ligand_serial[name] = serial
        out.append(_pdb_line(serial, name, ligand[name], element))
        serial += 1
    for left, right in BONDS:
        out.append(f"CONECT{ligand_serial[left]:5d}{ligand_serial[right]:5d}")
    out.append("END")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    return neighbors


def main() -> int:
    out_dir = HERE / "parents"
    for parent in PARENTS:
        src = RUN_ROOT / parent["pdb"]
        dest = out_dir / f"{parent['design']}_rfd3.pdb"
        neighbors = convert_one(src, dest)
        buried = sum(1 for count in neighbors.values() if count >= 3)
        print(f"{parent['design']} -> {dest.name} oxalate_neighbors={neighbors} buried_atoms={buried}/6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
