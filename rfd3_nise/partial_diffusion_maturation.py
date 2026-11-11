#!/usr/bin/env python
"""Affinity maturation for the oxaliplatin binders, following Wang et al. 2026.

Acta Pharm. Sin. B, doi:10.1016/j.apsb.2026.08.031. That paper moved a
NECTIN-4 minibinder from 262 nmol/L to 8.74 nmol/L by partial diffusion of
the three best complexes, not by single-site polarity edits. Each parent was
noised at Partial_T of 11, 13, 15, 21, 23 and 25 (800 scaffolds each),
redesigned, and kept only when the interface stayed confident.

Here the three parents are the oxaliplatin designs that already passed the
geometric gate. The ligand stays fixed. Sequence design uses LASErMPNN,
because ProteinMPNN does not see the platinum complex. Admission uses the
paper's interface cut adapted to Boltz: protein pLDDT at least 0.90, interface
PAE below 5.5, ipTM at least 0.80, ligand RMSD under 2.5 A, oxalate buried,
distal ring exposed. p(bind) does not admit a design. This module writes the
plan and does not launch RFdiffusion or NISE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

PARTIAL_T = (11, 13, 15, 21, 23, 25)
SCAFFOLDS_PER_T = 800
SEQUENCES_PER_BACKBONE = 3
SHORTLIST_PER_PARENT = 20
PROTEIN_PLDDT_MIN = 0.90
IPTM_MIN = 0.80
INTERFACE_PAE_MAX = 5.5
LIGAND_RMSD_MAX = 2.5

PARENTS = (
    {
        "design": "seq_57",
        "iteration": 31,
        "pdb": (
            "sampled_backbones/iter_31/boltz_results_boltz_inputs/predictions/"
            "chunk_0_seq_57/chunk_0_seq_57_model_0.pdb"
        ),
    },
    {
        "design": "seq_67",
        "iteration": 29,
        "pdb": (
            "sampled_backbones/iter_29/boltz_results_boltz_inputs/predictions/"
            "chunk_0_seq_67/chunk_0_seq_67_model_0.pdb"
        ),
    },
    {
        "design": "seq_126",
        "iteration": 17,
        "pdb": (
            "sampled_backbones/iter_17/boltz_results_boltz_inputs/predictions/"
            "chunk_0_seq_126/chunk_0_seq_126_model_0.pdb"
        ),
    },
)

# Site-3 analogue: keep the oxalate buried and the distal ring exposed, and
# let the pocket also touch the proximal ring carbons C3/C4.
HOTSPOTS = "C1,C2,O1,O2,O3,O4,C3,C4"
BURIED = "C1,C2,O1,O2,O3,O4"
EXPOSED = "C5,C6,C7,C8"
HBOND_ACCEPTORS = "O3,O4"


def passes_interface_filter(row: dict) -> bool:
    """Paper-style interface gate. A high p(bind) cannot replace it.

    Maturation in Wang et al. kept designs with interface PAE below 5.5 and
    binder pLDDT above 90, then ranked the survivors. ipTM is the second cut,
    matching their AlphaFold3 shortlist threshold of 0.8.
    """
    try:
        ligand_rmsd = float(row["ligand_rmsd"])
        protein_plddt = float(row["protein_plddt"])
        iptm = float(row["iptm"])
        interface_pae = float(row["interface_pae"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        ligand_rmsd < LIGAND_RMSD_MAX
        and protein_plddt >= PROTEIN_PLDDT_MIN
        and iptm >= IPTM_MIN
        and interface_pae < INTERFACE_PAE_MAX
        and bool(row.get("oxalate_buried"))
        and bool(row.get("dach_exposed"))
    )


def rank_shortlist(rows: list[dict], per_parent: int = SHORTLIST_PER_PARENT) -> list[dict]:
    """Keep the paper gate, then rank by interface PAE and ipTM. Cap each parent at 20."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if not passes_interface_filter(row):
            continue
        grouped.setdefault(str(row.get("parent", "")), []).append(row)
    ranked: list[dict] = []
    for parent in sorted(grouped):
        ordered = sorted(
            grouped[parent],
            key=lambda item: (float(item["interface_pae"]), -float(item["iptm"]), float(item["ligand_rmsd"])),
        )
        ranked.extend(ordered[:per_parent])
    return ranked


def render_jobs() -> str:
    """One Foundry job per parent and Partial_T. Only the ligand coordinates stay fixed.

    Foundry otherwise fixes every atom read from the input, and partial diffusion
    then returns the parent. partial_t is angstroms of noise on the unfixed protein.
    """
    lines = [
        "# Partial diffusion of the three oxaliplatin complexes that passed the gate.",
        "# partial_t is angstroms of noise in the installed Foundry RFD3. The values",
        "# 11, 13, 15, 21, 23, 25 and 800 scaffolds per value follow Wang et al. 2026,",
        "# doi:10.1016/j.apsb.2026.08.031. Inputs are RFD3-form parents written by",
        "# prepare_parents.py. select_fixed_atoms is the ligand only.",
        "",
    ]
    for parent in PARENTS:
        for partial_t in PARTIAL_T:
            name = f"oxp_{parent['design']}_t{partial_t}"
            lines.extend(
                [
                    f"{name}:",
                    f"  input: parents/{parent['design']}_rfd3.pdb",
                    "  ligand: l:g",
                    f"  partial_t: {partial_t}",
                    "  select_fixed_atoms: C1",
                    "  redesign_motif_sidechains: false",
                    "  infer_ori_strategy: hotspots",
                    "  is_non_loopy: true",
                    "  select_hotspots:",
                    f"    C1: {HOTSPOTS}",
                    "  select_buried:",
                    f"    C1: {BURIED}",
                    "  select_exposed:",
                    f"    C1: {EXPOSED}",
                    "  select_hbond_acceptor:",
                    f"    C1: {HBOND_ACCEPTORS}",
                    "",
                ]
            )
    return "\n".join(lines)


def build_plan() -> dict:
    n_jobs = len(PARENTS) * len(PARTIAL_T)
    return {
        "source": "10.1016/j.apsb.2026.08.031",
        "method": "partial_diffusion",
        "launch": False,
        "parents": list(PARENTS),
        "partial_t": list(PARTIAL_T),
        "scaffolds_per_t": SCAFFOLDS_PER_T,
        "n_jobs": n_jobs,
        "n_scaffolds": n_jobs * SCAFFOLDS_PER_T,
        "sequences_per_backbone": SEQUENCES_PER_BACKBONE,
        "sequence_model": "LASErMPNN",
        "rescore": "Boltz-2x",
        "rank_by": ["interface_pae", "iptm"],
        "pbind_does_not_admit": True,
        "filter": {
            "ligand_rmsd_max": LIGAND_RMSD_MAX,
            "protein_plddt_min": PROTEIN_PLDDT_MIN,
            "iptm_min": IPTM_MIN,
            "interface_pae_max": INTERFACE_PAE_MAX,
            "oxalate_buried": True,
            "dach_exposed": True,
            "shortlist_per_parent": SHORTLIST_PER_PARENT,
        },
        "paper_result": "Binder-33 partial diffusion changed 9 of 23 interface residues and improved BLI Kd from 262 nmol/L to 8.74 nmol/L. That gain was measured, not predicted.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the oxaliplatin partial-diffusion plan. Does not launch it.")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    (out / "maturation_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (out / "partial_diffusion_jobs.yaml").write_text(render_jobs(), encoding="utf-8")
    print(f"[oxp-mature] {plan['n_scaffolds']} scaffolds across {plan['n_jobs']} jobs, launch={plan['launch']}")
    print(f"[oxp-mature] wrote {out / 'maturation_plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
