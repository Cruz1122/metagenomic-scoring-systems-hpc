#!/usr/bin/env python3
"""Valida dataset generado por data/scripts/generate_data.py."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "csv"
NPY = ROOT / "npy"

samples = pd.read_csv(CSV / "samples.csv")
A_csv = pd.read_csv(CSV / "matrix_A.csv")
metadata = pd.read_csv(CSV / "metadata.csv")
functional = pd.read_csv(CSV / "functional_matrix.csv")
item_profiles = pd.read_csv(CSV / "item_profiles.csv")
A = np.load(NPY / "matrix_A.npy")
y = np.load(NPY / "labels.npy")
TSF = np.load(NPY / "profiles_TSF.npy")

assert samples.shape == (100, 3), samples.shape
assert int((samples["label"] == 0).sum()) == 50
assert int((samples["label"] == 1).sum()) == 50
assert A.shape == (100, 500), A.shape
assert y.shape == (100,), y.shape
assert TSF.shape == (500, 3), TSF.shape
assert len(functional) == 500
assert len(item_profiles) == 500
assert list(samples["sample_id"]) == list(A_csv["sample_id"])
assert list(samples["sample_id"]) == list(metadata["sample_id"])
assert "taxon_direction" in item_profiles.columns
np.testing.assert_allclose(A.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)
np.testing.assert_array_equal(samples["label"].to_numpy(np.int32), y)
assert np.all(TSF >= 0) and np.all(TSF <= 1)
print("OK: dataset integrity validated")
print(f"A: {A.shape}  labels: healthy={(y==0).sum()}, CRC={(y==1).sum()}  TSF: {TSF.shape}")
