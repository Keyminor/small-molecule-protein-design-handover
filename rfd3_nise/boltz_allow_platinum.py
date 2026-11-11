#!/root/autodl-tmp/NISE/.venv/bin/python
"""Boltz CLI that still scores platinum complexes.

Boltz affinity standardization uses ChEMBL's exclude_flag, which drops every
metal. Intact oxaliplatin is a SMILES RDKit already parses, so that exclusion
is skipped and the original SMILES is kept. Organic ligands still go through
Boltz's normal standardizer.
"""

from __future__ import annotations

import torch.fx._symbolic_trace as _fx_trace

# Torch 2.8 names this guard is_fx_tracing. The installed cuEquivariance build
# looks up the newer names and otherwise refuses to load the fast kernels.
if not hasattr(_fx_trace, "is_fx_symbolic_tracing"):
    _fx_trace.is_fx_symbolic_tracing = _fx_trace.is_fx_tracing
if not hasattr(_fx_trace, "is_fx_tracing_symbolic_tracing"):
    _fx_trace.is_fx_tracing_symbolic_tracing = _fx_trace.is_fx_tracing

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# This RDKit build no longer re-exports Descriptors on AllChem. Boltz affinity
# still calls AllChem.Descriptors.MolWt.
if not hasattr(AllChem, "Descriptors"):
    AllChem.Descriptors = Descriptors

import boltz.data.parse.schema as schema

_standardize = schema.standardize


def standardize(smiles: str) -> str:
    try:
        return _standardize(smiles)
    except ValueError as exc:
        if "excluded" not in str(exc):
            raise
        if Chem.MolFromSmiles(smiles) is None:
            raise
        return smiles


schema.standardize = standardize

from boltz.main import cli  # noqa: E402


if __name__ == "__main__":
    cli()
