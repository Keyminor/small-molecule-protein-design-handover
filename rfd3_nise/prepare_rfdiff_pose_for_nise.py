#!/usr/bin/env python
"""Convert an RFdiffusion3 complex PDB into a NISE-ready pose.

RFD3 typically writes the designed binder as chain A, an optional protein
target as chain B, and the ligand as chain C. NISE requires:

  * protein on chain A
  * exactly one ligand residue on chain B, resnum 1
  * protonated ligand + CONECT records
  * element-sequential atom names (C1, O1, ...)

This script remaps chains, drops solvent/ions, then reuses
protonate_and_add_conect_records.main() without changing that function.
Pharmacophore SMARTS from a constraints JSON are mapped onto the prepared
atom names and written beside the output PDB.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import prody as pr
from rdkit import Chem
from rdkit.Chem import AllChem

from _nise_root import nise_root
NISE_DIRECTORY_PATH = nise_root()
sys.path.insert(0, str(NISE_DIRECTORY_PATH))

from protonate_and_add_conect_records import main as protonate_input_pdb

WATER_RESNAMES = {"HOH", "WAT", "H2O", "SOL", "TIP", "T3P", "T4P"}
ION_RESNAMES = {
    "NA", "CL", "K", "MG", "ZN", "CA", "MN", "FE", "CU", "NI", "CO",
    "PO4", "SO4", "FMT", "ACE", "NH2",
}
DEFAULT_CONSTRAINTS_PATH = (
    NISE_DIRECTORY_PATH / "campaigns" / "eribulin" / "ligand" / "atom_constraints.json"
)


def _unique_resnames(atoms: pr.AtomGroup) -> set[str]:
    return {str(name).strip() for name in atoms.getResnames()}


def select_ligand_atoms(
    structure: pr.AtomGroup,
    ligand_resname: str | None = None,
    ligand_chain: str | None = None,
) -> pr.AtomGroup:
    hetero = structure.select("(not protein) and not element H")
    if hetero is None:
        raise ValueError("No non-protein heavy atoms found in the input PDB.")

    keep_sele = []
    for atom in hetero:
        resname = str(atom.getResname()).strip()
        if resname in WATER_RESNAMES or resname in ION_RESNAMES:
            continue
        if ligand_chain is not None and str(atom.getChid()).strip() != ligand_chain:
            continue
        if ligand_resname is not None and resname != ligand_resname:
            continue
        keep_sele.append(atom.getIndex())

    if not keep_sele:
        raise ValueError(
            "No ligand heavy atoms remained after dropping water/ions. "
            "Pass --ligand-resname / --ligand-chain if the ligand was filtered out."
        )

    ligand = structure.select("index " + " ".join(str(i) for i in keep_sele))
    if ligand is None:
        raise ValueError("Failed to select ligand atoms.")
    ligand = ligand.copy()

    resnames = _unique_resnames(ligand)
    if len(resnames) != 1:
        raise ValueError(
            f"Expected exactly one ligand residue name after cleaning, found {sorted(resnames)}. "
            "Remove extra hetero atoms or pass --ligand-resname."
        )
    if ligand.numResidues() != 1:
        raise ValueError(
            f"Expected exactly one ligand residue, found {ligand.numResidues()}."
        )
    return ligand


def remap_protein_to_chain_a(
    structure: pr.AtomGroup,
    protein_chains: Iterable[str] | None = None,
) -> pr.AtomGroup:
    protein = structure.select("protein")
    if protein is None:
        raise ValueError("No protein atoms found in the input PDB.")

    if protein_chains:
        wanted = {chain.strip() for chain in protein_chains if chain.strip()}
        keep = [atom.getIndex() for atom in protein if str(atom.getChid()).strip() in wanted]
        if not keep:
            raise ValueError(f"No protein atoms found on chains {sorted(wanted)}.")
        protein = structure.select("index " + " ".join(str(i) for i in keep))
        if protein is None:
            raise ValueError("Failed to select requested protein chains.")

    protein = protein.copy()
    protein.setChids("A")
    return protein


def write_cleaned_complex(
    protein: pr.AtomGroup,
    ligand: pr.AtomGroup,
    output_path: Path,
    ligand_3lc: str,
) -> Path:
    ligand = ligand.copy()
    ligand.setResnames(ligand_3lc)
    ligand.setChids("C")
    ligand.setResnums(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pr.writePDB(str(output_path), protein + ligand)
    return output_path


def _atom_index_to_pdb_name(pdb_mol: Chem.Mol) -> dict[int, str]:
    mapping = {}
    for atom in pdb_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        mapping[atom.GetIdx()] = info.GetName().strip()
    return mapping


def _expand_atom_indices(mol: Chem.Mol, seed: Iterable[int], n_bonds: int) -> set[int]:
    if n_bonds <= 0:
        return set(seed)
    visited = set(seed)
    queue = deque((idx, 0) for idx in seed)
    while queue:
        idx, dist = queue.popleft()
        if dist >= n_bonds:
            continue
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in visited:
                continue
            visited.add(nidx)
            queue.append((nidx, dist + 1))
    return visited


def map_constraint_groups(
    ligand_pdb_block: str,
    smiles: str,
    constraints: dict[str, Any],
) -> dict[str, list[str]]:
    """Map SMARTS-defined groups onto prepared ligand atom names."""
    # Protonated NISE PDBs include hydrogens. Distance-inferred H bonds make
    # carbons look pentavalent; match topology on heavy atoms only.
    pdb_mol = Chem.MolFromPDBBlock(ligand_pdb_block, removeHs=True, sanitize=True)
    smi_mol = Chem.MolFromSmiles(smiles)
    if pdb_mol is None or smi_mol is None:
        raise ValueError("RDKit could not parse the prepared ligand PDB or the SMILES.")

    try:
        pdb_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
    except Exception as exc:
        raise ValueError(
            "SMILES does not match the prepared ligand geometry "
            f"(AssignBondOrdersFromTemplate failed): {exc}"
        ) from exc

    query_mol = Chem.AddHs(smi_mol)
    name_by_idx = _atom_index_to_pdb_name(pdb_mol)
    resolved: dict[str, list[str]] = {}

    for group_name, spec in constraints.get("groups", {}).items():
        smarts = spec.get("smarts")
        if not smarts:
            continue
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f"Invalid SMARTS for group {group_name}: {smarts}")

        matches = pdb_mol.GetSubstructMatches(query)
        if not matches:
            # Try the hydrogen-stripped copy used as the SMARTS target.
            matches = query_mol.GetSubstructMatches(query)
            if matches:
                raise ValueError(
                    f"SMARTS for {group_name} matched the SMILES but not the prepared PDB. "
                    "Check protonation / atom completeness."
                )
            raise ValueError(f"SMARTS for group {group_name} did not match the prepared ligand.")

        seed = matches[0]
        expand = int(spec.get("expand_bonds", 0) or 0)
        atom_indices = _expand_atom_indices(pdb_mol, seed, expand)
        names = sorted({name_by_idx[i] for i in atom_indices if i in name_by_idx and not name_by_idx[i].startswith("H")})
        if not names:
            raise ValueError(f"Group {group_name} resolved to no heavy-atom names.")
        resolved[group_name] = names

    return resolved


def roles_from_constraints(
    constraints: dict[str, Any],
    resolved_groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    roles = {"rmsd_mask": set(), "buried": set(), "exposed": set()}
    for group_name, names in resolved_groups.items():
        role = constraints.get("groups", {}).get(group_name, {}).get("role")
        if role not in roles:
            continue
        roles[role].update(names)
    return {role: sorted(names) for role, names in roles.items()}


def extract_ligand_pdb_block(prepared_pdb_path: Path) -> str:
    structure = pr.parsePDB(str(prepared_pdb_path))
    ligand = structure.select("not protein")
    if ligand is None:
        raise ValueError(f"Prepared PDB has no ligand: {prepared_pdb_path}")
    stream = io.StringIO()
    pr.writePDBStream(stream, ligand)
    return stream.getvalue()


def validate_prepared_pose(prepared_pdb_path: Path, smiles: str) -> None:
    structure = pr.parsePDB(str(prepared_pdb_path))
    protein = structure.select("protein")
    ligand = structure.select("not protein")
    if protein is None or ligand is None:
        raise ValueError("Prepared PDB must contain protein and ligand.")

    protein_chains = {str(c).strip() for c in protein.getChids()}
    if protein_chains != {"A"}:
        raise ValueError(f"Prepared protein must be chain A only, found {sorted(protein_chains)}.")

    ligand_chains = {str(c).strip() for c in ligand.getChids()}
    if ligand_chains != {"B"}:
        raise ValueError(f"Prepared ligand must be chain B only, found {sorted(ligand_chains)}.")
    if ligand.numResidues() != 1:
        raise ValueError(f"Prepared ligand must be one residue, found {ligand.numResidues()}.")

    text = prepared_pdb_path.read_text(encoding="utf-8", errors="replace")
    if "CONECT" not in text:
        raise ValueError("Prepared PDB is missing CONECT records.")

    ligand_block = extract_ligand_pdb_block(prepared_pdb_path)
    pdb_mol = Chem.MolFromPDBBlock(ligand_block)
    smi_mol = Chem.MolFromSmiles(smiles)
    if pdb_mol is None or smi_mol is None:
        raise ValueError("RDKit failed to parse prepared ligand or SMILES during validation.")
    try:
        AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
    except Exception as exc:
        pdb_heavy = pdb_mol.GetNumHeavyAtoms()
        smi_heavy = smi_mol.GetNumHeavyAtoms()
        raise ValueError(
            "Prepared ligand does not match SMILES. "
            f"PDB heavy atoms={pdb_heavy}, SMILES heavy atoms={smi_heavy}. {exc}"
        ) from exc


def prepare_rfdiff_pose(
    input_pdb: Path,
    output_pdb: Path,
    smiles: str,
    ligand_3lc: str = "ERB",
    protein_chains: list[str] | None = None,
    ligand_resname: str | None = None,
    ligand_chain: str | None = None,
    constraints_path: Path | None = None,
    constraints_output: Path | None = None,
) -> dict[str, Any]:
    structure = pr.parsePDB(str(input_pdb))
    if structure is None:
        raise ValueError(f"Could not parse PDB: {input_pdb}")

    protein = remap_protein_to_chain_a(structure, protein_chains)
    ligand = select_ligand_atoms(structure, ligand_resname=ligand_resname, ligand_chain=ligand_chain)

    cleaned_path = output_pdb.with_name(output_pdb.stem + "_cleaned_pre_protonate.pdb")
    write_cleaned_complex(protein, ligand, cleaned_path, ligand_3lc)

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    protonate_input_pdb(str(cleaned_path), str(output_pdb), smiles)
    validate_prepared_pose(output_pdb, smiles)

    manifest: dict[str, Any] = {
        "input_pdb": str(input_pdb),
        "cleaned_pdb": str(cleaned_path),
        "prepared_pdb": str(output_pdb),
        "ligand_smiles": smiles,
        "ligand_3lc": ligand_3lc,
        "resolved_groups": {},
        "roles": {"rmsd_mask": [], "buried": [], "exposed": []},
    }

    if constraints_path is not None:
        constraints = json.loads(Path(constraints_path).read_text(encoding="utf-8"))
        ligand_block = extract_ligand_pdb_block(output_pdb)
        resolved = map_constraint_groups(ligand_block, smiles, constraints)
        roles = roles_from_constraints(constraints, resolved)
        manifest["resolved_groups"] = resolved
        manifest["roles"] = roles
        manifest["constraints_path"] = str(constraints_path)

        if constraints_output is None:
            constraints_output = output_pdb.with_name(output_pdb.stem + "_prepared_atom_constraints.json")
        constraints_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest["prepared_constraints_path"] = str(constraints_output)

    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert an RFdiffusion3 pose into a NISE-ready PDB.",
    )
    p.add_argument("--input-pdb", type=Path, required=True, help="RFD3 (or RFD3-like) complex PDB.")
    p.add_argument("--output-pdb", type=Path, required=True, help="Prepared NISE PDB path.")
    p.add_argument("--smiles", type=str, default=None, help="Ligand SMILES. Defaults to Eribulin free base.")
    p.add_argument("--ligand-3lc", type=str, default="ERB")
    p.add_argument("--protein-chains", type=str, default=None, help="Optional protein chain IDs, e.g. 'A'.")
    p.add_argument("--ligand-resname", type=str, default=None)
    p.add_argument("--ligand-chain", type=str, default=None, help="Optional source ligand chain, e.g. 'C'.")
    p.add_argument("--constraints-json", type=Path, default=DEFAULT_CONSTRAINTS_PATH)
    p.add_argument("--no-constraints", action="store_true")
    p.add_argument("--constraints-output", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smiles:
        smiles = args.smiles
    else:
        from campaigns.eribulin.constants import load_ligand_smiles
        smiles = load_ligand_smiles()

    protein_chains = None
    if args.protein_chains:
        protein_chains = [tok for tok in args.protein_chains.replace(",", " ").split() if tok]

    constraints_path = None if args.no_constraints else args.constraints_json
    if constraints_path is not None and not constraints_path.exists():
        print(f"[prepare] constraints file not found, skipping mapping: {constraints_path}")
        constraints_path = None

    manifest = prepare_rfdiff_pose(
        input_pdb=args.input_pdb.expanduser().resolve(),
        output_pdb=args.output_pdb.expanduser().resolve(),
        smiles=smiles,
        ligand_3lc=args.ligand_3lc,
        protein_chains=protein_chains,
        ligand_resname=args.ligand_resname,
        ligand_chain=args.ligand_chain,
        constraints_path=constraints_path,
        constraints_output=args.constraints_output,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "resolved_groups"}, indent=2))
    if manifest.get("resolved_groups"):
        print("[prepare] resolved pharmacophore atom names:")
        for group, names in manifest["resolved_groups"].items():
            print(f"  {group}: {' '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
