#!/usr/bin/env python
"""Wrapper around an external RFdiffusion3 / Foundry `rfd3 design` CLI.

RFD3 lives in its own environment. This script only prepares a ligand-only
input PDB + YAML and optionally shells out to `rfd3`. It does not install
Foundry into the NISE venv and does not touch the NISE selection loop.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

from _nise_root import nise_root
NISE_DIRECTORY_PATH = nise_root()
sys.path.insert(0, str(NISE_DIRECTORY_PATH))

from campaigns.eribulin.constants import (
    BIOACTIVE_RFD3_PDB,
    DEFAULT_BINDER_LENGTH,
    DEFAULT_LIGAND_CONTIG,
    LIGAND_SDF_PATH,
    RFD3_JOB_RECOMMENDED_YAML_PATH,
    RFD3_SCALE_JSON_PATH,
    RFD3_YAML_TEMPLATE_PATH,
    load_ligand_smiles,
)

SCALE_PRESET_NAMES = ("smoke", "medium", "campaign", "precise", "diverse", "xlarge", "pocket")

# CCD "ERB" is erythromycin B, not eribulin. Foundry docs say a colon in the
# residue name skips CCD lookup for custom ligands. NISE still remaps to ERB.
RFD3_LIGAND_RESNAME = "l:g"

WATER_RESNAMES = {"HOH", "WAT", "H2O", "SOL", "TIP", "T3P", "T4P"}
ION_RESNAMES = {
    "NA", "CL", "K", "MG", "ZN", "CA", "MN", "FE", "CU", "NI", "CO",
    "PO4", "SO4", "FMT", "ACE", "NH2",
}


def _require_rdkit():
    from rdkit import Chem
    return Chem


def ligand_sdf_to_rfd3_pdb(sdf_path: Path, output_pdb: Path, ligand_3lc: str = RFD3_LIGAND_RESNAME) -> Path:
    """Write the reference ligand as chain C residue 1 for RFD3 contig C1-1."""
    Chem = _require_rdkit()
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"Could not read a molecule from {sdf_path}")
    mol = Chem.RemoveHs(mol)

    pdb_block = Chem.MolToPDBBlock(mol)
    lines = []
    atom_serial = 1
    for line in pdb_block.splitlines():
        if line.startswith("HETATM") or line.startswith("ATOM"):
            # Columns: ATOM/HETATM name lives in 13-16, resname 18-20, chain 22, resseq 23-26.
            rec = "HETATM"
            rest = line[6:]
            name = line[12:16]
            element = line[76:78] if len(line) >= 78 else name.strip()[:2]
            coords = line[30:54]
            occupancy = line[54:60] if len(line) >= 60 else "  1.00"
            temp = line[60:66] if len(line) >= 66 else "  0.00"
            rewritten = (
                f"{rec}{atom_serial:5d} {name} {ligand_3lc:>3s} C{1:4d}    "
                f"{coords}{occupancy}{temp}          {element:>2s}"
            )
            lines.append(rewritten)
            atom_serial += 1
        elif line.startswith("CONECT"):
            lines.append(line)
    if not lines:
        raise ValueError(f"No atoms written from {sdf_path}")

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\nEND\n")
    return output_pdb


def clean_existing_complex_pdb(input_pdb: Path, output_pdb: Path) -> Path:
    """Drop water/ions from a user-supplied RFD3 input complex."""
    import prody as pr

    structure = pr.parsePDB(str(input_pdb))
    keep_idx = []
    for atom in structure:
        resname = str(atom.getResname()).strip()
        if resname in WATER_RESNAMES or resname in ION_RESNAMES:
            continue
        keep_idx.append(atom.getIndex())
    if not keep_idx:
        raise ValueError(f"Cleaning {input_pdb} removed every atom.")
    cleaned = structure.select("index " + " ".join(str(i) for i in keep_idx)).copy()
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    pr.writePDB(str(output_pdb), cleaned)
    return output_pdb


def _atom_names_from_pdb_lines(ligand_pdb: Path) -> list[str]:
    """Parse HETATM/ATOM names without ProDy (used after SDF conversion)."""
    names = []
    for line in ligand_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not (line.startswith("HETATM") or line.startswith("ATOM")):
            continue
        name = line[12:16].strip()
        element = line[76:78].strip() if len(line) >= 78 else "".join(ch for ch in name if ch.isalpha())
        if element.upper() == "H" or name.upper().startswith("H"):
            continue
        names.append(name)
    return names


def load_pharmacophore() -> dict | None:
    path = RFD3_YAML_TEMPLATE_PATH.parent / "pharmacophore.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ligand_hotspot_atom_names(ligand_pdb: Path, max_atoms: int = 12) -> list[str]:
    """Pick a small set of non-hydrogen ligand atom names for select_hotspots."""
    try:
        import prody as pr
    except ImportError:
        names = _atom_names_from_pdb_lines(ligand_pdb)
    else:
        structure = pr.parsePDB(str(ligand_pdb))
        lig = structure.select("(not protein) and not element H")
        if lig is None:
            lig = structure.select("not element H")
        if lig is None:
            return []
        names = [str(name).strip() for name in lig.getNames()]
    # Prefer oxygen / carbonyl-like atoms first, then the rest.
    oxygens = [n for n in names if n.startswith("O")]
    carbons = [n for n in names if n.startswith("C")]
    ordered = oxygens + carbons + [n for n in names if n not in oxygens and n not in carbons]
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique[:max_atoms]


def write_rfd3_yaml(
    template_path: Path,
    output_path: Path,
    input_pdb: Path,
    contig: str,
    hotspot_atoms: list[str],
    job_name: str = "eribulin_ligand_only",
    buried_atoms: list[str] | None = None,
    exposed_atoms: list[str] | None = None,
    hbond_acceptors: list[str] | None = None,
) -> Path:
    del template_path  # Kept in the signature so callers can pass the campaign template.

    def _csv(atoms: list[str] | None) -> str:
        return ",".join(atoms or [])

    lines = [
        f"{job_name}:",
        f"  input: {input_pdb}",
        f"  ligand: {RFD3_LIGAND_RESNAME}",
        f"  contig: {contig}",
        "  redesign_motif_sidechains: false",
        "  infer_ori_strategy: hotspots",
        "  is_non_loopy: true",
        "  select_hotspots:",
    ]
    if hotspot_atoms:
        lines.append(f"    C1: {_csv(hotspot_atoms)}")
    if buried_atoms:
        lines.extend(["  select_buried:", f"    C1: {_csv(buried_atoms)}"])
    if exposed_atoms:
        lines.extend(["  select_exposed:", f"    C1: {_csv(exposed_atoms)}"])
    if hbond_acceptors:
        lines.extend(["  select_hbond_acceptor:", f"    C1: {_csv(hbond_acceptors)}"])
    text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def load_scale_config(path: Path | None = None) -> dict:
    scale_path = path or RFD3_SCALE_JSON_PATH
    return json.loads(scale_path.read_text(encoding="utf-8"))


def named_yaml_jobs(yaml_text: str) -> list[str]:
    names: list[str] = []
    for line in yaml_text.splitlines():
        if not line or line.startswith("#") or line.startswith(" ") or line.startswith("\t"):
            continue
        if line.endswith(":"):
            names.append(line[:-1].strip())
    return names


def parse_job_keys(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    keys = [tok.strip() for tok in raw.split(",") if tok.strip()]
    return keys or None


def rewrite_yaml_input_paths(src: Path, dest: Path) -> Path:
    """Copy a jobs YAML and resolve relative `input:` paths against the source file."""
    src_dir = src.parent.resolve()
    out_lines: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(\s*input:\s+)(.+)$", line)
        if match:
            raw = match.group(2).strip().strip("'").strip('"')
            path = Path(raw)
            if not path.is_absolute():
                path = (src_dir / path).resolve()
            line = f"{match.group(1)}{path.as_posix()}"
        out_lines.append(line)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return dest


def first_yaml_field(yaml_text: str, field: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(field)}:\s*(.+)$", yaml_text, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip().strip("'").strip('"')


def resolve_scale_params(
    *,
    scale: str | None = None,
    n_batches: int | None = None,
    diffusion_batch_size: int | None = None,
    n_designs: int | None = None,
    num_timesteps: int | None = None,
    noise_scale: float | None = None,
    step_scale: float | None = None,
    s_trans: float | None = None,
    scale_config: dict | None = None,
) -> dict:
    """Map RFD1/2-style knobs onto Foundry RFD3 CLI values.

    Design count is n_batches * diffusion_batch_size per named YAML job.
    If the batch size is lowered (OOM), n_batches is recomputed from n_designs.
    """
    presets = (scale_config or load_scale_config()).get("presets", {})
    preset = presets.get(scale or "", {})
    batch_size = (
        diffusion_batch_size if diffusion_batch_size is not None else int(preset.get("diffusion_batch_size", 8))
    )
    target = n_designs if n_designs is not None else preset.get("n_designs")
    if n_batches is not None:
        batches = int(n_batches)
        resolved_designs = batches * batch_size
    elif target is not None:
        batches = max(1, math.ceil(int(target) / batch_size))
        resolved_designs = batches * batch_size
    else:
        batches = 1
        resolved_designs = batches * batch_size
    return {
        "scale": scale,
        "n_batches": int(batches),
        "diffusion_batch_size": int(batch_size),
        "n_designs": int(resolved_designs),
        "num_timesteps": num_timesteps if num_timesteps is not None else preset.get("num_timesteps"),
        "noise_scale": noise_scale if noise_scale is not None else preset.get("noise_scale"),
        "step_scale": step_scale if step_scale is not None else preset.get("step_scale"),
        "s_trans": s_trans if s_trans is not None else preset.get("s_trans"),
    }


def sampler_cli_args(
    resolved: dict,
    *,
    dump_trajectories: bool = False,
    low_memory_mode: bool = False,
    job_keys: list[str] | None = None,
) -> list[str]:
    args: list[str] = []
    if resolved.get("num_timesteps") is not None:
        args.append(f"inference_sampler.num_timesteps={resolved['num_timesteps']}")
    if resolved.get("step_scale") is not None:
        args.append(f"inference_sampler.step_scale={resolved['step_scale']}")
    if resolved.get("noise_scale") is not None:
        args.append(f"inference_sampler.noise_scale={resolved['noise_scale']}")
    if resolved.get("s_trans") is not None:
        args.append(f"inference_sampler.s_trans={resolved['s_trans']}")
    if dump_trajectories:
        args.append("dump_trajectories=True")
    if low_memory_mode:
        args.append("low_memory_mode=True")
    if job_keys:
        args.append("json_keys_subset=[" + ",".join(job_keys) + "]")
    return args


def build_rfd3_command(
    executable: str,
    out_dir: Path,
    inputs_yaml: Path,
    n_batches: int,
    diffusion_batch_size: int,
    extra_args: list[str],
) -> list[str]:
    command = [
        executable,
        "design",
        f"out_dir={out_dir}",
        f"inputs={inputs_yaml}",
        f"n_batches={n_batches}",
        f"diffusion_batch_size={diffusion_batch_size}",
    ]
    command.extend(extra_args)
    return command


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare and optionally run RFdiffusion3 for Eribulin.")
    p.add_argument("--rfd3-executable", default="rfd3", help="Foundry RFD3 CLI (must be on PATH or absolute).")
    p.add_argument("--ligand-sdf", type=Path, default=LIGAND_SDF_PATH)
    p.add_argument("--input-pdb", type=Path, default=None, help="Optional prebuilt RFD3 input complex.")
    p.add_argument(
        "--inputs-yaml",
        type=Path,
        default=None,
        help="Use an existing Foundry jobs YAML instead of generating one from the SDF.",
    )
    p.add_argument(
        "--from-sdf",
        action="store_true",
        help="Build a ligand-only PDB from --ligand-sdf (ETKDG). Default is the 5JH7 bioactive job_recommended.yaml.",
    )
    p.add_argument(
        "--job-keys",
        default=None,
        help="Comma-separated YAML job names to run (maps to json_keys_subset). Use this to avoid multiplying n_batches by every job in jobs.yaml.",
    )
    p.add_argument("--output-dir", type=Path, default=NISE_DIRECTORY_PATH / "campaigns" / "eribulin" / "rfd3_outputs")
    p.add_argument("--binder-length", default=DEFAULT_BINDER_LENGTH, help="Designed binder length contig, e.g. 150-200.")
    p.add_argument(
        "--split-contig",
        action="store_true",
        help="Design two disconnected segments (60-100,/0,60-100,/0,C1-1) instead of one chain.",
    )
    p.add_argument("--ligand-contig", default=DEFAULT_LIGAND_CONTIG)
    p.add_argument("--scale", choices=list(SCALE_PRESET_NAMES), default=None, help="Design-count / sampler preset from rfd3/scale.json.")
    p.add_argument("--n-designs", type=int, default=None, help="Target designs per YAML job. Sets n_batches = ceil(n_designs / diffusion_batch_size).")
    p.add_argument("--n-batches", type=int, default=None)
    p.add_argument("--diffusion-batch-size", type=int, default=None)
    p.add_argument("--num-timesteps", type=int, default=None, help="RFD3 inference_sampler.num_timesteps (Foundry default 200).")
    p.add_argument("--noise-scale", type=float, default=None, help="RFD3 inference_sampler.noise_scale (Foundry default 1.003). Try 0.8 or 1.2 for alternate series.")
    p.add_argument("--step-scale", type=float, default=None, help="RFD3 inference_sampler.step_scale (Foundry default 1.5). Higher = less diverse.")
    p.add_argument("--s-trans", type=float, default=None, help="RFD3 inference_sampler.s_trans translational noise (Foundry default 1.0).")
    p.add_argument("--dump-trajectories", action="store_true", help="Save diffusion trajectories. Do not use at campaign scale.")
    p.add_argument("--low-memory-mode", action="store_true")
    p.add_argument("--fill-hotspots", action="store_true", default=True)
    p.add_argument("--no-fill-hotspots", action="store_false", dest="fill_hotspots")
    p.add_argument("--dry-run", action="store_true", help="Write YAML + cleaned input; do not call rfd3.")
    p.add_argument("rfd3_extra", nargs=argparse.REMAINDER, help="Extra rfd3 design key=value args after --")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cli = argv if argv is not None else sys.argv[1:]
    explicit_sdf = bool(args.from_sdf or "--ligand-sdf" in cli)
    if (
        args.inputs_yaml is None
        and args.input_pdb is None
        and not explicit_sdf
        and RFD3_JOB_RECOMMENDED_YAML_PATH.exists()
        and BIOACTIVE_RFD3_PDB.exists()
    ):
        args.inputs_yaml = RFD3_JOB_RECOMMENDED_YAML_PATH

    resolved = resolve_scale_params(
        scale=args.scale,
        n_batches=args.n_batches,
        diffusion_batch_size=args.diffusion_batch_size,
        n_designs=args.n_designs,
        num_timesteps=args.num_timesteps,
        noise_scale=args.noise_scale,
        step_scale=args.step_scale,
        s_trans=args.s_trans,
    )
    job_keys = parse_job_keys(args.job_keys)
    yaml_path = output_dir / "rfd3_inputs.yaml"
    buried_atoms: list[str] = []
    exposed_atoms: list[str] = []
    hbond_acceptors: list[str] = []
    hotspot_atoms: list[str] = []

    if args.inputs_yaml is not None:
        src_yaml = args.inputs_yaml.expanduser().resolve()
        if not src_yaml.exists():
            raise SystemExit(f"inputs YAML not found: {src_yaml}")
        rewrite_yaml_input_paths(src_yaml, yaml_path)
        yaml_text = yaml_path.read_text(encoding="utf-8")
        contig = first_yaml_field(yaml_text, "contig") or f"{args.binder_length},/0,{args.ligand_contig}"
        cleaned_pdb = Path(first_yaml_field(yaml_text, "input") or "")
        hotspot_csv = first_yaml_field(yaml_text, "C1")
        if hotspot_csv:
            hotspot_atoms = [tok for tok in hotspot_csv.split(",") if tok]
    else:
        if args.input_pdb is not None:
            cleaned_pdb = output_dir / "rfd3_input_cleaned.pdb"
            clean_existing_complex_pdb(args.input_pdb.expanduser().resolve(), cleaned_pdb)
        else:
            sdf_path = args.ligand_sdf.expanduser().resolve()
            if not sdf_path.exists():
                raise SystemExit(
                    f"Ligand SDF not found: {sdf_path}\n"
                    "Run campaigns/eribulin/ligand/build_reference_sdf.py "
                    "or pass --ligand-sdf / --input-pdb / --inputs-yaml."
                )
            cleaned_pdb = output_dir / "eribulin_ligand_only.pdb"
            ligand_sdf_to_rfd3_pdb(sdf_path, cleaned_pdb)

        pharmacophore = load_pharmacophore() if args.fill_hotspots else None
        if pharmacophore and pharmacophore.get("rfd3"):
            rfd3_sel = pharmacophore["rfd3"]
            hotspot_atoms = [tok for tok in rfd3_sel.get("select_hotspots", {}).get("C1", "").split(",") if tok]
            buried_atoms = [tok for tok in rfd3_sel.get("select_buried", {}).get("C1", "").split(",") if tok]
            exposed_atoms = [tok for tok in rfd3_sel.get("select_exposed", {}).get("C1", "").split(",") if tok]
            hbond_acceptors = [tok for tok in rfd3_sel.get("select_hbond_acceptor", {}).get("C1", "").split(",") if tok]
        else:
            hotspot_atoms = ligand_hotspot_atom_names(cleaned_pdb) if args.fill_hotspots else []
        if args.split_contig:
            contig = f"60-100,/0,60-100,/0,{args.ligand_contig}"
        else:
            contig = f"{args.binder_length},/0,{args.ligand_contig}"
        write_rfd3_yaml(
            RFD3_YAML_TEMPLATE_PATH,
            yaml_path,
            cleaned_pdb,
            contig,
            hotspot_atoms,
            buried_atoms=buried_atoms,
            exposed_atoms=exposed_atoms,
            hbond_acceptors=hbond_acceptors,
        )

    yaml_text = yaml_path.read_text(encoding="utf-8")
    job_names = named_yaml_jobs(yaml_text)
    active_jobs = job_keys if job_keys else job_names
    n_jobs = len(active_jobs) if active_jobs else 1
    total_designs = n_jobs * resolved["n_batches"] * resolved["diffusion_batch_size"]
    extra = [tok for tok in args.rfd3_extra if tok != "--"]
    sampler = sampler_cli_args(
        resolved,
        dump_trajectories=bool(args.dump_trajectories),
        low_memory_mode=bool(args.low_memory_mode),
        job_keys=job_keys,
    )
    command = build_rfd3_command(
        args.rfd3_executable,
        output_dir,
        yaml_path,
        resolved["n_batches"],
        resolved["diffusion_batch_size"],
        sampler + extra,
    )

    manifest = {
        "cleaned_input_pdb": str(cleaned_pdb),
        "yaml": str(yaml_path),
        "contig": contig,
        "hotspot_atoms": hotspot_atoms,
        "buried_atoms": buried_atoms,
        "exposed_atoms": exposed_atoms,
        "hbond_acceptors": hbond_acceptors,
        "command": command,
        "dry_run": bool(args.dry_run),
        "ligand_smiles": load_ligand_smiles(),
        "scale": resolved["scale"],
        "n_batches": resolved["n_batches"],
        "diffusion_batch_size": resolved["diffusion_batch_size"],
        "n_designs_per_job": resolved["n_designs"],
        "n_jobs": n_jobs,
        "job_names": active_jobs,
        "total_designs": total_designs,
        "num_timesteps": resolved["num_timesteps"],
        "noise_scale": resolved["noise_scale"],
        "step_scale": resolved["step_scale"],
    }
    (output_dir / "rfd3_launch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[rfd3] wrote cleaned input {cleaned_pdb}")
    print(f"[rfd3] wrote YAML {yaml_path}")
    print(
        f"[rfd3] expected designs: {n_jobs} job(s) × {resolved['n_batches']} batches × "
        f"{resolved['diffusion_batch_size']} = {total_designs}"
    )
    if n_jobs > 1 and total_designs >= 1000:
        print(
            "[rfd3] WARNING: multiple YAML jobs multiply the design count. "
            "Pass --inputs-yaml campaigns/eribulin/rfd3/job_recommended.yaml "
            "or --job-keys eribulin_recommended_150_200."
        )
    print("[rfd3] command:", " ".join(str(tok) for tok in command))

    if args.dry_run:
        print("[rfd3] dry-run: not invoking RFdiffusion3.")
        return 0

    if shutil.which(args.rfd3_executable) is None and not Path(args.rfd3_executable).exists():
        raise SystemExit(
            f"RFD3 executable not found: {args.rfd3_executable}. "
            "Install Foundry RFD3 in a separate environment and pass --rfd3-executable."
        )

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
