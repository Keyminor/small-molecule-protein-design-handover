"""Fail closed before a 14400-scaffold launch.

Confirms one converted parent parses as Foundry partial diffusion, the ligand
stays in the atom array, and partial_t=11 cuts the noise schedule instead of
keeping every step.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAML = HERE / "partial_diffusion_jobs.yaml"
PARENT = HERE / "parents" / "seq_57_rfd3.pdb"


def _noise_kept(partial_t: float, num_timesteps: int = 300) -> tuple[int, float, float]:
    import torch
    from rfd3.model.inference_sampler import SampleDiffusionWithMotif

    sampler = SampleDiffusionWithMotif(num_timesteps=num_timesteps)
    schedule = sampler._construct_inference_noise_schedule(
        device=torch.device("cpu"),
        partial_t=torch.tensor([partial_t]),
    )
    return int(schedule.numel()), float(schedule.min()), float(schedule.max())


def _load_spec():
    import yaml
    from rfd3.inference.input_parsing import DesignInputSpecification

    jobs = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    spec = jobs["oxp_seq_57_t11"]
    spec["input"] = str(PARENT)
    return DesignInputSpecification(**spec)


def main() -> int:
    if not PARENT.exists():
        print(f"missing {PARENT}", file=sys.stderr)
        return 1
    kept, lo, hi = _noise_kept(11.0)
    full, _, full_hi = _noise_kept(1.0e9)
    print(f"schedule partial_t=11 keeps {kept}/{full} steps, range [{lo:.3f}, {hi:.3f}], full_max={full_hi:.1f}")
    if kept == 0 or kept >= full:
        print("partial_t did not cut the schedule", file=sys.stderr)
        return 1
    spec = _load_spec()
    atom_array = spec.build()
    names = [str(name).strip() for name in atom_array.res_name]
    ligand_mask = [name in {"l:g", "L:G"} for name in names]
    ligand = sum(ligand_mask)
    protein = sum(not flag for flag in ligand_mask)
    chains = sorted({str(c).strip() for c in atom_array.chain_id})
    fixed = None
    protein_fixed = None
    if "is_motif_atom_with_fixed_coord" in atom_array.get_annotation_categories():
        flags = atom_array.get_annotation("is_motif_atom_with_fixed_coord")
        fixed = int(sum(int(flag) for flag, is_lig in zip(flags, ligand_mask) if is_lig))
        protein_fixed = int(sum(int(flag) for flag, is_lig in zip(flags, ligand_mask) if not is_lig))
    print(
        f"atoms={len(atom_array)} ligand_atoms={ligand} ligand_fixed={fixed} "
        f"protein_atoms={protein} protein_fixed={protein_fixed} chains={chains}"
    )
    if ligand < 15:
        print("ligand was dropped from the partial-diffusion input", file=sys.stderr)
        return 1
    if fixed != ligand:
        print("ligand coordinates are not fixed", file=sys.stderr)
        return 1
    if protein_fixed != 0:
        print("protein coordinates are fixed, so partial diffusion copies the parent", file=sys.stderr)
        return 1
    if "partial_t" not in atom_array.get_annotation_categories():
        print("partial_t annotation missing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
