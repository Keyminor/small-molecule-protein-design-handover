"""Convert oxaliplatin RFD3 cif.gz outputs and prepare them for Boltz / NISE.

Each design cif contains the ligand twice (chain B and chain C). Preparation
keeps chain C, residue l:g, and renames it to OXP.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import prody as pr

HERE = Path(__file__).resolve().parent
from _nise_root import nise_root
NISE = nise_root()
if str(NISE) not in sys.path:
    sys.path.insert(0, str(NISE))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from campaigns.oxaliplatin.constants import ATOM_CONSTRAINTS_PATH, load_ligand_smiles
from prepare_rfdiff_pose_for_nise import main as prepare_main

SRC = NISE / "campaigns" / "oxaliplatin" / "rfd3_outputs_pocket"
PDB_DIR = NISE / "campaigns" / "oxaliplatin" / "pocket_run" / "rfd3"
PREP_DIR = NISE / "campaigns" / "oxaliplatin" / "pocket_run" / "prepared"


def main() -> int:
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    smiles = load_ligand_smiles()
    n_conv = n_prep = n_fail = 0
    gz_files = sorted(SRC.glob("*.cif.gz"))
    for idx, gz in enumerate(gz_files, start=1):
        pdb = PDB_DIR / gz.name.replace(".cif.gz", ".pdb")
        dest = PREP_DIR / pdb.name
        cst = dest.with_name(dest.stem + "_prepared_atom_constraints.json")
        if dest.exists() and dest.stat().st_size > 1000 and cst.exists():
            continue
        try:
            if not pdb.exists() or pdb.stat().st_size < 1000:
                cif = PDB_DIR / gz.name.replace(".cif.gz", ".cif")
                if not cif.exists():
                    cif.write_bytes(gzip.open(gz, "rb").read())
                pr.writePDB(str(pdb), pr.parseMMCIF(str(cif)))
                cif.unlink(missing_ok=True)
                n_conv += 1
            rc = prepare_main([
                "--input-pdb", str(pdb),
                "--output-pdb", str(dest),
                "--smiles", smiles,
                "--ligand-3lc", "OXP",
                "--ligand-chain", "C",
                "--constraints-json", str(ATOM_CONSTRAINTS_PATH),
            ])
        except Exception as exc:
            rc = 1
            print("prepare exception", pdb.name, exc)
        if rc == 0:
            n_prep += 1
        else:
            n_fail += 1
            dest.unlink(missing_ok=True)
            cst.unlink(missing_ok=True)
            print("prepare failed", pdb.name, rc)
        if idx % 50 == 0 or idx == len(gz_files):
            ready = [
                p for p in PREP_DIR.glob("*.pdb")
                if "cleaned_pre_protonate" not in p.name
            ]
            print(
                f"progress {idx}/{len(gz_files)} converted={n_conv} "
                f"newly_prepared={n_prep} failed={n_fail} prepared_total={len(ready)}",
                flush=True,
            )
    ready = [p for p in PREP_DIR.glob("*.pdb") if "cleaned_pre_protonate" not in p.name]
    print(f"DONE converted={n_conv} newly_prepared={n_prep} failed={n_fail} prepared_total={len(ready)}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
