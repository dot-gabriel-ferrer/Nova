"""NOVA — Next-generation Open Volumetric Archive.

Reference Python implementation for the NOVA astronomical data format.
A cloud-native scientific data format designed to succeed FITS.
"""

__version__ = "0.1.0"

from nova.container import NovaDataset, open_dataset, create_dataset
from nova.wcs import NovaWCS
from nova.fits_converter import from_fits, to_fits
from nova.provenance import ProvenanceBundle
from nova.integrity import compute_sha256, verify_chunk
from nova.benchmarks import run_full_comparison, generate_test_data

__all__ = [
    "NovaDataset",
    "open_dataset",
    "create_dataset",
    "NovaWCS",
    "from_fits",
    "to_fits",
    "ProvenanceBundle",
    "compute_sha256",
    "verify_chunk",
    "run_full_comparison",
    "generate_test_data",
]
