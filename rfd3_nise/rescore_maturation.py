#!/usr/bin/env python
"""LASErMPNN redesign and Boltz-2x rescoring of partial-diffusion scaffolds.

Three sequences per backbone. Boltz is structure-only: the admission gate does
not use p(bind), so the affinity head is not run. Interface PAE is the mean of
the protein–ligand blocks in the Boltz PAE matrix. A missing PAE fails closed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from _nise_root import nise_root
REPO = nise_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from campaigns.oxaliplatin.constants import (  # noqa: E402
    BURIED_ATOMS,
    EXPOSED_ATOMS,
    RMSD_MASK,
    load_ligand_smiles,
)
from campaigns.oxaliplatin.partial_diffusion_maturation import (  # noqa: E402
    SEQUENCES_PER_BACKBONE,
    rank_shortlist,
)

SCAFFOLDS = HERE / "rfd3_outputs"
WORK = HERE
PARENT_RE = re.compile(r"oxp_(seq_\d+)_t\d+")
BURIAL_ALPHA = 9.0
BURIAL_RAYS = 5


def parent_from_name(name: str) -> str:
    match = PARENT_RE.search(name)
    if match is None:
        raise ValueError(f"no parent in {name}")
    return match.group(1)


def interface_pae(pae, n_protein: int, n_ligand: int) -> float:
    """Mean PAE between protein tokens and ligand tokens, both directions."""
    import numpy as np

    matrix = np.asarray(pae, dtype=float)
    expected = n_protein + n_ligand
    if n_protein < 1 or n_ligand < 1 or matrix.shape != (expected, expected):
        raise ValueError(f"PAE shape {matrix.shape} does not match {n_protein}+{n_ligand} tokens")
    protein_to_ligand = matrix[:n_protein, n_protein:]
    ligand_to_protein = matrix[n_protein:, :n_protein]
    return float((protein_to_ligand.mean() + ligand_to_protein.mean()) / 2.0)


def iter_cif_atoms(text: str):
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].startswith("loop_"):
            index += 1
            continue
        index += 1
        columns: list[str] = []
        while index < len(lines) and lines[index].startswith("_"):
            columns.append(lines[index].split()[0])
            index += 1
        if "_atom_site.Cartn_x" not in columns:
            while index < len(lines) and not lines[index].startswith(("loop_", "data_")):
                index += 1
            continue
        while index < len(lines) and lines[index] and not lines[index].startswith(("loop_", "#", "data_")):
            parts = lines[index].split()
            index += 1
            if len(parts) < len(columns):
                continue
            yield dict(zip(columns, parts))
        continue


def _pdb_line(serial: int, record: str, name: str, resn: str, chain: str, resseq: int, xyz: tuple[float, float, float], element: str) -> str:
    atom = f"{name:>4s}"[:4]
    residue = f"{resn:>3s}"[:3]
    return (
        f"{record}{serial:5d} {atom} {residue} {chain[:1]}{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}"
    )


def cif_text_to_pdb(text: str) -> str:
    """Write protein chain A and the ligand as PDB. Hydrogens are dropped."""
    rows: list[str] = []
    serial = 1
    last_chain = None
    for atom in iter_cif_atoms(text):
        element = atom.get("_atom_site.type_symbol", "").upper()
        if element in {"H", "D"}:
            continue
        chain = atom.get("_atom_site.label_asym_id", "A")[:1]
        if last_chain is not None and chain != last_chain:
            rows.append("TER")
        last_chain = chain
        record = "HETATM" if atom.get("_atom_site.group_PDB") == "HETATM" else "ATOM  "
        raw_seq = atom.get("_atom_site.auth_seq_id", ".")
        if raw_seq in {".", "?"}:
            raw_seq = atom["_atom_site.label_seq_id"]
        resseq = int(float(raw_seq))
        xyz = (
            float(atom["_atom_site.Cartn_x"]),
            float(atom["_atom_site.Cartn_y"]),
            float(atom["_atom_site.Cartn_z"]),
        )
        rows.append(
            _pdb_line(
                serial,
                record,
                atom["_atom_site.label_atom_id"],
                atom["_atom_site.label_comp_id"],
                chain,
                resseq,
                xyz,
                element,
            )
        )
        serial += 1
    if not rows:
        raise ValueError("cif contained no atoms")
    rows.append("END")
    return "\n".join(rows) + "\n"


def read_structure_text(path: Path) -> str:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def convert_scaffolds(src: Path, dest: Path, limit: int | None) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    scaffolds = sorted(src.glob("*.cif.gz"))
    if limit is not None:
        scaffolds = scaffolds[:limit]
    for index, path in enumerate(scaffolds, start=1):
        out = dest / f"{path.name.removesuffix('.cif.gz')}.pdb"
        if not out.exists() or out.stat().st_size < 1000:
            out.write_text(cif_text_to_pdb(read_structure_text(path)), encoding="utf-8")
        written.append(out)
        if index % 500 == 0 or index == len(scaffolds):
            print(f"[convert] {index}/{len(scaffolds)}", flush=True)
    return written


def _designs_ready(design_dir: Path, n_sequences: int) -> bool:
    return all((design_dir / f"design_{idx}.pdb").exists() for idx in range(n_sequences))


def run_laser(pdb_dir: Path, design_dir: Path, n_sequences: int, limit: int | None) -> None:
    design_dir.mkdir(parents=True, exist_ok=True)
    pdbs = sorted(pdb_dir.glob("*.pdb"))
    if limit is not None:
        pdbs = pdbs[:limit]
    pending = [path for path in pdbs if not _designs_ready(design_dir / path.stem, n_sequences)]
    print(f"[laser] pending {len(pending)} of {len(pdbs)}", flush=True)
    if not pending:
        return
    manifest = design_dir / "pending.txt"
    manifest.write_text("\n".join(str(path) for path in pending) + "\n", encoding="utf-8")
    script = REPO / "LASErMPNN" / "run_batch_inference.py"
    weights = REPO / "LASErMPNN" / "model_weights" / "laser_weights_0p1A_nothing_heldout.pt"
    command = [
        sys.executable,
        str(script),
        str(manifest),
        str(design_dir),
        str(n_sequences),
        "--device",
        "cuda:0",
        "--model_weights_path",
        str(weights),
        "--designs_per_batch",
        str(n_sequences),
        "--inputs_processed_simultaneously",
        "4",
        "--sequence_temp",
        "0.5",
        "--first_shell_sequence_temp",
        "0.7",
        "--chi_temp",
        "1e-6",
        "--disabled_residues",
        "X,C",
        "--constrain_ala_gly_sampling_to_exposed_non_secondary_structure",
        "--ala_budget",
        "4",
        "--gly_budget",
        "0",
        "--disable_charged_fs",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print("[laser]", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _sequence(pdb_path: Path) -> str:
    from rank_rfdiff_poses_with_boltz import protein_sequence_from_pdb

    sequence = protein_sequence_from_pdb(pdb_path)
    if "X" in sequence:
        raise ValueError(f"nonstandard residue in {pdb_path.name}")
    return sequence


def _yaml_text(sequence: str, smiles: str) -> str:
    return (
        "version: 1\nsequences:\n"
        f"  - protein:\n      id: A\n      sequence: {sequence}\n      msa: empty\n"
        f"  - ligand:\n      id: B\n      smiles: '{smiles}'\n"
    )


def _load_done(jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl.exists():
        return done
    for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("design"):
            done.add(str(row["design"]))
    return done


def _score_one(laser_pdb: Path, boltz_pdb: Path, pae_path: Path, smiles: str) -> dict:
    import numpy as np
    import prody as pr

    from utility_scripts.burial_calc import compute_fast_ligand_burial_mask
    from utility_scripts.calc_symmetry_aware_rmsd import _main as calc_rmsd

    laser = pr.parsePDB(str(laser_pdb))
    boltz = pr.parsePDB(str(boltz_pdb))
    protein_rmsd, ligand_rmsd, laser_to_boltz = calc_rmsd(
        laser,
        boltz,
        smiles,
        use_chirality=False,
        reference_names_to_ignore=set(RMSD_MASK),
        align_on_binding_site=True,
    )
    if not laser_to_boltz:
        raise ValueError("ligand atoms did not match the oxaliplatin template")
    boltz_to_laser = {value: key for key, value in laser_to_boltz.items()}
    protein = boltz.select("chid A and name CA")
    ligand = boltz.select("chid B and not element H")
    if protein is None or ligand is None:
        raise ValueError("Boltz output is missing protein or ligand")
    ligand.setNames([boltz_to_laser[name] for name in ligand.getNames()])
    n_protein = int(protein.numAtoms())
    n_ligand = int(ligand.numAtoms())
    with np.load(pae_path) as payload:
        pae = interface_pae(payload["pae"], n_protein, n_ligand)
    names = [str(name).strip() for name in ligand.getNames()]
    coords = ligand.getCoords()
    buried_mask = compute_fast_ligand_burial_mask(protein.getCoords(), coords, alpha=BURIAL_ALPHA, num_rays=BURIAL_RAYS)
    buried_flags = [bool(flag) for flag in buried_mask]

    def _role(atom_names: tuple[str, ...]) -> list[bool]:
        flags = []
        for atom_name in atom_names:
            if atom_name not in names:
                raise ValueError(f"missing ligand atom {atom_name}")
            flags.append(buried_flags[names.index(atom_name)])
        return flags

    oxalate = _role(tuple(BURIED_ATOMS))
    dach = _role(tuple(EXPOSED_ATOMS))
    conf_path = boltz_pdb.parent / f"confidence_{boltz_pdb.stem}.json"
    confidence = json.loads(conf_path.read_text(encoding="utf-8"))
    return {
        "ligand_rmsd": float(ligand_rmsd),
        "protein_rmsd": float(protein_rmsd),
        "protein_plddt": float(protein.getBetas().mean()) / 100.0,
        "iptm": float(confidence["iptm"]),
        "interface_pae": pae,
        "oxalate_buried": all(oxalate),
        "dach_exposed": not any(dach),
    }


def _find_prediction(out_dir: Path, stem: str) -> tuple[Path, Path] | None:
    matches = sorted(out_dir.rglob(f"{stem}_model_0.pdb"))
    if not matches:
        return None
    pdb_path = matches[0]
    pae_path = pdb_path.parent / f"pae_{pdb_path.stem}.npz"
    if not pae_path.exists():
        return None
    return pdb_path, pae_path


def run_boltz(design_dir: Path, work: Path, smiles: str, limit: int | None, chunk_size: int = 32) -> int:
    jsonl = work / "scores.jsonl"
    done = _load_done(jsonl)
    designs = sorted(design_dir.glob("*/design_*.pdb"))
    if limit is not None:
        designs = designs[: limit * SEQUENCES_PER_BACKBONE]
    pending = [path for path in designs if f"{path.parent.name}_{path.stem}" not in done]
    print(f"[boltz] pending {len(pending)} of {len(designs)}", flush=True)
    boltz_exec = HERE / "boltz_allow_platinum.py"
    cache = os.environ.get("BOLTZ_CACHE", "")
    kept = work / "kept"
    kept.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        chunk_id = start // chunk_size
        chunk_dir = work / f"_boltz_chunk_{chunk_id:05d}"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        inputs = chunk_dir / "inputs"
        inputs.mkdir(parents=True)
        stems: dict[str, Path] = {}
        for design in chunk:
            stem = f"{design.parent.name}_{design.stem}"
            sequence = _sequence(design)
            (inputs / f"{stem}.yaml").write_text(_yaml_text(sequence, smiles), encoding="utf-8")
            stems[stem] = design
        out_dir = chunk_dir / "out"
        command = [
            sys.executable,
            str(boltz_exec),
            "predict",
            str(inputs),
            "--devices",
            "1",
            "--out_dir",
            str(out_dir),
            "--output_format",
            "pdb",
            "--sampling_steps",
            "200",
            "--use_potentials",
        ]
        if cache:
            command.extend(["--cache", cache])
        print(f"[boltz] chunk {chunk_id} n={len(chunk)}", flush=True)
        completed = subprocess.run(command)
        if completed.returncode != 0:
            print(f"[boltz] chunk {chunk_id} failed with {completed.returncode}", flush=True)
            shutil.rmtree(chunk_dir, ignore_errors=True)
            return completed.returncode
        with jsonl.open("a", encoding="utf-8") as handle:
            for stem, design in stems.items():
                found = _find_prediction(out_dir, stem)
                row = {
                    "parent": parent_from_name(stem),
                    "design": stem,
                    "laser_pdb": str(design),
                }
                if found is None:
                    row["error"] = "missing_boltz_output"
                else:
                    boltz_pdb, pae_path = found
                    try:
                        row.update(_score_one(design, boltz_pdb, pae_path, smiles))
                    except Exception as exc:
                        row["error"] = str(exc)
                    else:
                        from campaigns.oxaliplatin.partial_diffusion_maturation import passes_interface_filter

                        row["passes"] = passes_interface_filter(row)
                        if row["passes"]:
                            shutil.copy2(boltz_pdb, kept / boltz_pdb.name)
                            row["kept_boltz_pdb"] = str(kept / boltz_pdb.name)
                handle.write(json.dumps(row) + "\n")
                print(
                    f"[boltz] {stem} pass={row.get('passes', False)} "
                    f"iptm={row.get('iptm')} pae={row.get('interface_pae')} rmsd={row.get('ligand_rmsd')}",
                    flush=True,
                )
        shutil.rmtree(chunk_dir, ignore_errors=True)
    return 0


def write_shortlist(work: Path) -> Path:
    jsonl = work / "scores.jsonl"
    rows = []
    if not jsonl.exists():
        raise FileNotFoundError(f"no scores at {jsonl}")
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    ranked = rank_shortlist(rows)
    dest = work / "shortlist.json"
    payload = {
        "n_scored": len(rows),
        "n_shortlist": len(ranked),
        "pbind_does_not_admit": True,
        "designs": ranked,
    }
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[shortlist] {len(ranked)} of {len(rows)} -> {dest}", flush=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Redesign and rescore partial-diffusion scaffolds.")
    parser.add_argument("stage", choices=("convert", "laser", "boltz", "shortlist", "all"))
    parser.add_argument("--scaffolds", type=Path, default=SCAFFOLDS)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--limit", type=int, default=0, help="Number of scaffolds. 0 means all.")
    args = parser.parse_args(argv)
    limit = args.limit or None
    work = args.work.expanduser().resolve()
    pdb_dir = work / "backbones_pdb"
    design_dir = work / "laser_designs"
    smiles = load_ligand_smiles()
    if args.stage in {"convert", "all"}:
        convert_scaffolds(args.scaffolds.expanduser().resolve(), pdb_dir, limit)
    if args.stage in {"laser", "all"}:
        run_laser(pdb_dir, design_dir, SEQUENCES_PER_BACKBONE, limit)
    if args.stage in {"boltz", "all"}:
        status = run_boltz(design_dir, work, smiles, limit)
        if status != 0:
            return status
    if args.stage in {"shortlist", "all"}:
        write_shortlist(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
