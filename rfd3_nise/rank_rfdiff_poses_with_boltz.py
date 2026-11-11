#!/usr/bin/env python
"""Rank prepared RFD3 poses with Boltz and keep a single NISE backbone.

NISE DesignCampaign accepts exactly one PDB in input_backbones/. This script
scores many prepared poses, then copies only the best one into that directory.
The NISE selection loop is not modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import prody as pr

from _nise_root import nise_root
NISE_DIRECTORY_PATH = nise_root()
sys.path.insert(0, str(NISE_DIRECTORY_PATH))

from campaigns.eribulin.constants import (
    DEFAULT_OBJECTIVE_FUNCTION,
    load_ligand_smiles,
)


def get_boltz_yaml_boilerplate(sequence: str, smiles: str, predict_affinity: bool) -> str:
    """Local copy of the NISE Boltz YAML header; keeps this script import-light."""
    boltz_yaml = (
        "version: 1\nsequences:\n"
        f"  - protein:\n      id: A\n      sequence: {sequence}\n      msa: empty\n"
        f"  - ligand:\n      id: B\n      smiles: '{smiles}'\n"
    )
    if predict_affinity:
        boltz_yaml += "properties:\n  - affinity:\n      binder: B\n"
    return boltz_yaml

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def protein_sequence_from_pdb(pdb_path: Path) -> str:
    structure = pr.parsePDB(str(pdb_path))
    calphas = structure.select("protein and name CA")
    if calphas is None:
        raise ValueError(f"No protein CA atoms in {pdb_path}")
    sequence = []
    for resname in calphas.getResnames():
        key = str(resname).strip().upper()
        sequence.append(AA3_TO_1.get(key, "X"))
    if not sequence:
        raise ValueError(f"Empty protein sequence in {pdb_path}")
    return "".join(sequence)


def write_pose_yaml(prepared_pdb: Path, smiles: str, yaml_path: Path, predict_affinity: bool) -> Path:
    sequence = protein_sequence_from_pdb(prepared_pdb)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(get_boltz_yaml_boilerplate(sequence, smiles, predict_affinity), encoding="utf-8")
    return yaml_path


def ligand_plddt_from_boltz_pdb(pdb_path: Path) -> float:
    structure = pr.parsePDB(str(pdb_path))
    ligand = structure.select("chid B and not element H")
    if ligand is None:
        ligand = structure.select("(not protein) and not element H")
    if ligand is None:
        return float("nan")
    return float(ligand.getBetas().mean() / 100.0)


def collect_boltz_outputs(boltz_output_dir: Path, yaml_stem: str) -> dict:
    predictions = boltz_output_dir / "boltz_results_boltz_inputs" / "predictions" / yaml_stem
    if not predictions.exists():
        # Some Boltz versions nest one extra directory.
        matches = list(boltz_output_dir.rglob(f"{yaml_stem}_model_0.pdb"))
        pdb_path = matches[0] if matches else None
        pred_dir = pdb_path.parent if pdb_path else predictions
    else:
        pdb_path = predictions / f"{yaml_stem}_model_0.pdb"
        pred_dir = predictions

    record = {
        "yaml_stem": yaml_stem,
        "boltz_pdb": str(pdb_path) if pdb_path and pdb_path.exists() else None,
        "iptm": None,
        "pbind": None,
        "ligand_plddt": None,
        "score": None,
    }
    if pdb_path is None or not pdb_path.exists():
        return record

    conf_path = pred_dir / f"confidence_{pdb_path.stem}.json"
    if conf_path.exists():
        confidence = json.loads(conf_path.read_text(encoding="utf-8"))
        record["iptm"] = confidence.get("iptm")

    affinity_path = pred_dir / f"affinity_{yaml_stem}.json"
    if affinity_path.exists():
        affinity = json.loads(affinity_path.read_text(encoding="utf-8"))
        record["pbind"] = affinity.get("affinity_probability_binary")

    record["ligand_plddt"] = ligand_plddt_from_boltz_pdb(pdb_path)
    pbind = record["pbind"] if record["pbind"] is not None else 0.0
    plddt = record["ligand_plddt"] if record["ligand_plddt"] is not None else 0.0
    record["score"] = float(plddt) + float(pbind)
    return record


def rank_records(records: list[dict]) -> list[dict]:
    def sort_key(item: dict):
        score = item.get("score")
        return (score is not None, score if score is not None else float("-inf"))
    return sorted(records, key=sort_key, reverse=True)


def copy_best_to_nise_input(best_prepared_pdb: Path, nise_input_dir: Path) -> Path:
    dest_dir = nise_input_dir / "input_backbones"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for old in dest_dir.glob("*.pdb"):
        old.unlink()
    dest = dest_dir / best_prepared_pdb.name
    shutil.copy2(best_prepared_pdb, dest)
    constraints = best_prepared_pdb.with_name(best_prepared_pdb.stem + "_prepared_atom_constraints.json")
    if constraints.exists():
        shutil.copy2(constraints, dest_dir / constraints.name)
    return dest


def _chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0 or size >= len(items):
        return [items] if items else []
    return [items[i : i + size] for i in range(0, len(items), size)]


def _cleanup_pose_predictions(output_dir: Path, stem: str) -> None:
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        if stem not in path.name:
            continue
        if path.suffix in {".npz", ".pkl", ".h5", ".cif"} or path.name.endswith(".cif.gz"):
            path.unlink(missing_ok=True)
        elif path.suffix == ".pdb" and "model_" in path.name:
            path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rank prepared RFD3 poses with Boltz and pick one NISE seed.")
    p.add_argument("--prepared-dir", type=Path, required=True, help="Directory of prepared NISE PDBs.")
    p.add_argument("--output-dir", type=Path, required=True, help="Boltz work directory + ranking outputs.")
    p.add_argument("--nise-input-dir", type=Path, default=None, help="Campaign dir that will contain input_backbones/.")
    p.add_argument("--smiles", type=str, default=None)
    p.add_argument("--boltz-executable", type=Path, default=NISE_DIRECTORY_PATH / ".venv" / "bin" / "boltz")
    p.add_argument("--boltz-devices", nargs="+", default=["cuda:0"])
    p.add_argument("--sampling-steps", type=int, default=200)
    p.add_argument("--chunk-size", type=int, default=0, help="Boltz YAML batch size. 0 = all at once.")
    p.add_argument("--cleanup-predictions", action="store_true", help="Delete bulky Boltz arrays after scoring a chunk.")
    p.add_argument("--boltz-cache", type=Path, default=None)
    p.add_argument("--objective-function", default=DEFAULT_OBJECTIVE_FUNCTION)
    p.add_argument("--dry-run", action="store_true", help="Write Boltz YAMLs and ranking scaffold only.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared_dir = args.prepared_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    smiles = args.smiles or load_ligand_smiles()
    predict_affinity = "pbind" in args.objective_function

    prepared_pdbs = sorted(p for p in prepared_dir.glob("*.pdb") if "cleaned_pre_protonate" not in p.name)
    if not prepared_pdbs:
        raise SystemExit(f"No prepared PDBs found in {prepared_dir}")

    boltz_inputs = output_dir / "boltz_inputs"
    boltz_inputs.mkdir(parents=True, exist_ok=True)
    yaml_to_prepared = {}
    for pdb_path in prepared_pdbs:
        yaml_path = boltz_inputs / f"{pdb_path.stem}.yaml"
        write_pose_yaml(pdb_path, smiles, yaml_path, predict_affinity)
        yaml_to_prepared[pdb_path.stem] = pdb_path
        print(f"[rank] wrote {yaml_path}")

    ranking_path = output_dir / "pose_ranking.json"
    if args.dry_run:
        records = [
            {
                "yaml_stem": stem,
                "prepared_pdb": str(path),
                "iptm": None,
                "pbind": None,
                "ligand_plddt": None,
                "score": None,
            }
            for stem, path in yaml_to_prepared.items()
        ]
        ranking_path.write_text(json.dumps({"dry_run": True, "records": records}, indent=2) + "\n", encoding="utf-8")
        print(f"[rank] dry-run: wrote {ranking_path}; not calling Boltz.")
        return 0

    # Call Boltz directly. Do not import run_nise_boltz2x here: that pulls in
    # LASErMPNN / torch_scatter and can fail on older host glibc.
    device_ints = [tok.split(":")[-1] for tok in args.boltz_devices]
    cache = args.boltz_cache
    if cache is None:
        env_cache = os.environ.get("BOLTZ_CACHE")
        cache = Path(env_cache) if env_cache else None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(device_ints)
    stems = list(yaml_to_prepared)
    records_by_stem: dict[str, dict] = {}
    for chunk_idx, chunk in enumerate(_chunked(stems, int(args.chunk_size)), start=1):
        if int(args.chunk_size) > 0:
            chunk_dir = output_dir / f"_chunk_{chunk_idx:04d}"
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir)
            chunk_dir.mkdir(parents=True)
            for stem in chunk:
                src = boltz_inputs / f"{stem}.yaml"
                dest = chunk_dir / src.name
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                dest.symlink_to(src)
            predict_dir = chunk_dir
            print(f"[rank] chunk {chunk_idx}: {len(chunk)} yaml(s)")
        else:
            predict_dir = boltz_inputs
        command = [
            str(Path(args.boltz_executable).expanduser().resolve()),
            "predict",
            str(predict_dir),
            "--devices",
            str(len(device_ints)),
            "--out_dir",
            str(output_dir),
            "--output_format",
            "pdb",
            "--override",
            "--sampling_steps",
            str(int(args.sampling_steps)),
            "--sampling_steps_affinity",
            str(int(args.sampling_steps)),
            "--use_potentials",
            "--no_kernels",
        ]
        if cache is not None:
            command.extend(["--cache", str(Path(cache).expanduser().resolve())])
        print("[rank]", " ".join(command))
        subprocess.run(command, check=False, env=env)
        for stem in chunk:
            record = collect_boltz_outputs(output_dir, stem)
            record["prepared_pdb"] = str(yaml_to_prepared[stem])
            records_by_stem[stem] = record
            if args.cleanup_predictions:
                _cleanup_pose_predictions(output_dir, stem)
        if int(args.chunk_size) > 0:
            shutil.rmtree(predict_dir, ignore_errors=True)
        for leftover in output_dir.glob("boltz_results_*"):
            shutil.rmtree(leftover, ignore_errors=True)
        ranking_path.write_text(
            json.dumps({"dry_run": False, "records": rank_records(list(records_by_stem.values()))}, indent=2) + "\n",
            encoding="utf-8",
        )
    records = [records_by_stem[stem] for stem in stems]

    ranked = rank_records(records)
    ranking_path.write_text(json.dumps({"dry_run": False, "records": ranked}, indent=2) + "\n", encoding="utf-8")
    print(f"[rank] wrote {ranking_path}")

    scored = [row for row in ranked if row.get("score") is not None]
    if not scored:
        raise SystemExit("Boltz produced no scorable poses. Inspect boltz outputs before launching NISE.")

    best = scored[0]
    nise_input_dir = args.nise_input_dir or (output_dir / "nise_run")
    dest = copy_best_to_nise_input(Path(best["prepared_pdb"]), Path(nise_input_dir))
    print(f"[rank] selected {best['prepared_pdb']} -> {dest} (score={best['score']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
