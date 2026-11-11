#!/usr/bin/env python
"""Oxaliplatin NISE campaign launcher.

Imports the unchanged DesignCampaign / main() from run_nise_boltz2x.py and
only overrides campaign parameters. Use a single prepared PDB in
<input-dir>/input_backbones/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from _nise_root import nise_root
NISE_DIRECTORY_PATH = nise_root()
LASER_PATH = NISE_DIRECTORY_PATH / "LASErMPNN"
sys.path.insert(0, str(NISE_DIRECTORY_PATH))

from campaigns.oxaliplatin.constants import (
    DEFAULT_BURIAL_ALPHA,
    DEFAULT_LIGAND_RMSD_THRESHOLD,
    DEFAULT_OBJECTIVE_FUNCTION,
    DEFAULT_PROTEIN_RMSD_THRESHOLD,
    LIGAND_3LC,
    load_ligand_smiles,
)


def _load_role_atoms(constraints_path: Path | None) -> dict[str, set]:
    roles = {"rmsd_mask": set(), "buried": set(), "exposed": set()}
    if constraints_path is None or not constraints_path.exists():
        return roles
    payload = json.loads(constraints_path.read_text(encoding="utf-8"))
    stored = payload.get("roles") or {}
    for role in roles:
        roles[role] = set(stored.get(role, []))
    return roles


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run NISE on a prepared oxaliplatin pose.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=NISE_DIRECTORY_PATH / "campaigns" / "oxaliplatin" / "nise_run",
        help="Campaign directory that already contains input_backbones/ with exactly one PDB.",
    )
    p.add_argument("--smiles", type=str, default=None)
    p.add_argument("--prepared-constraints", type=Path, default=None)
    p.add_argument("--boltz-devices", nargs="+", default=["cuda:0"])
    p.add_argument("--debug", action="store_true")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num-iterations", type=int, default=35)
    p.add_argument("--sequences-sampled-per-backbone", type=int, default=None)
    return p


def build_params(args: argparse.Namespace) -> dict:
    smiles = args.smiles or load_ligand_smiles()
    constraints_path = args.prepared_constraints
    if constraints_path is None:
        candidates = list((args.input_dir / "input_backbones").glob("*prepared_atom_constraints.json"))
        if len(candidates) == 1:
            constraints_path = candidates[0]
    roles = _load_role_atoms(constraints_path)

    laser_sampling_params = {
        "sequence_temp": 0.5,
        "first_shell_sequence_temp": 0.7,
        "chi_temp": 1e-6,
        "seq_min_p": 0.0,
        "chi_min_p": 0.0,
        "disable_pbar": True,
        "disabled_residues_list": ["X", "C"],
        "constrain_ala_gly_sampling_to_exposed_non_secondary_structure": True,
        "budget_residue_sele_string": None,
        "ala_budget": 4,
        "gly_budget": 0,
        "disable_charged_fs": True,
    }

    return dict(
        debug=args.debug,
        use_wandb=args.use_wandb,
        input_dir=args.input_dir.expanduser().resolve(),
        ligand_3lc=LIGAND_3LC,
        ligand_rmsd_mask_atoms=roles["rmsd_mask"],
        ligand_atoms_enforce_buried=roles["buried"],
        ligand_atoms_enforce_exposed=roles["exposed"],
        laser_sampling_params=laser_sampling_params,
        ligand_smiles=smiles,
        objective_function=DEFAULT_OBJECTIVE_FUNCTION,
        drop_rmsd_mask_atoms_from_ligand_plddt_calc=True,
        keep_input_backbone_in_queue=False,
        keep_best_generator_backbone=True,
        rmsd_use_chirality=False,
        self_consistency_ligand_rmsd_threshold=DEFAULT_LIGAND_RMSD_THRESHOLD,
        self_consistency_protein_rmsd_threshold=DEFAULT_PROTEIN_RMSD_THRESHOLD,
        align_on_binding_site=True,
        fixed_identity_residue_indices=None,
        use_reduce_protonation=False,
        reduce_hetdict_path=None,
        reduce_executable_path=None,
        model_checkpoint=LASER_PATH / "model_weights" / "laser_weights_0p1A_nothing_heldout.pt",
        num_iterations=args.num_iterations,
        num_top_backbones_per_round=3,
        sequences_sampled_at_once=30,
        boltz2x_executable_path=str(
            (Path(__file__).resolve().parent / "boltz_allow_platinum.py").resolve()
        ),
        boltz2_cache_directory=os.environ.get("BOLTZ_CACHE") or None,
        boltz2_sampling_steps=200,
        boltz_inference_devices=list(args.boltz_devices),
        use_boltz_conformer_potentials=True,
        boltz2_predict_affinity=True,
        use_boltz_1x=False,
        boltz2_disable_kernels=True,
        boltz2_disable_nccl_p2p=False,
        sequences_sampled_per_backbone=(
            args.sequences_sampled_per_backbone
            if args.sequences_sampled_per_backbone is not None
            else (64 if not args.debug else 1 * len(args.boltz_devices))
        ),
        burial_mask_alpha_hull_alpha=DEFAULT_BURIAL_ALPHA,
        laser_inference_device=args.boltz_devices[0],
        laser_inference_dropout=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params = build_params(args)
    print("[oxaliplatin] resolved NISE params (loop unchanged):")
    printable = {k: (sorted(v) if isinstance(v, set) else str(v)) for k, v in params.items()}
    print(json.dumps(printable, indent=2, default=str))
    if args.dry_run:
        print("[oxaliplatin] dry-run: not launching DesignCampaign.")
        return 0
    from run_nise_boltz2x import main as nise_main
    nise_main(**params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
