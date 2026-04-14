"""NOVA -- Next-generation Open Volumetric Archive.

Reference Python implementation for the NOVA astronomical data format.
A cloud-native scientific data format designed to succeed FITS.
"""

from nova.constants import NOVA_VERSION

__version__ = NOVA_VERSION

from nova.container import NovaDataset, NovaExtension, NovaTable, open_dataset, create_dataset
from nova.wcs import NovaWCS
from nova.fits_converter import from_fits, to_fits
from nova.provenance import ProvenanceBundle
from nova.integrity import compute_sha256, verify_chunk
from nova.benchmarks import run_full_comparison, run_multi_format_comparison, generate_test_data
from nova.fast_io import fast_write, fast_read
from nova.validation import validate_metadata, validate_wcs, validate_provenance, validate_store
from nova.ml import (
    NormalizationMetadata,
    compute_normalization,
    normalize,
    denormalize,
    to_tensor,
)
from nova.remote import open_remote, is_remote_url
from nova.migrate import migrate_directory, MigrationReport
from nova.streaming import StreamWriter, open_appendable, append_frame
from nova.adapters import to_ccddata, from_ccddata, to_nddata, nova_to_hdulist
from nova.pipeline import Pipeline, Step, PipelineLog, StepLog
from nova.operations import (
    OperationHistory,
    OperationRecord,
    op_add,
    op_subtract,
    op_multiply,
    op_divide,
    op_clip,
    op_mask_replace,
    op_normalize,
    op_rebin,
    op_combine,
)
from nova import constants as constants
from nova import math as math
from nova import visualization as viz
from nova import image_processing as image_processing
from nova import photometry as photometry
from nova import spectral as spectral
from nova import coords as coords
from nova import catalog as catalog
from nova import pipeline as pipeline
from nova import operations as operations
from nova import astrometry as astrometry
from nova import photometry_pipeline as photometry_pipeline
from nova import spectroscopy_pipeline as spectroscopy_pipeline

# Convenience aliases matching README examples
open = open_dataset
validate = validate_store

__all__ = [
    # Core
    "NovaDataset",
    "NovaExtension",
    "NovaTable",
    "open_dataset",
    "create_dataset",
    "open",
    # WCS
    "NovaWCS",
    # FITS converter
    "from_fits",
    "to_fits",
    # Provenance
    "ProvenanceBundle",
    # Integrity
    "compute_sha256",
    "verify_chunk",
    # Benchmarks
    "run_full_comparison",
    "run_multi_format_comparison",
    "generate_test_data",
    # Fast I/O
    "fast_write",
    "fast_read",
    # Validation
    "validate_metadata",
    "validate_wcs",
    "validate_provenance",
    "validate_store",
    "validate",
    # ML
    "NormalizationMetadata",
    "compute_normalization",
    "normalize",
    "denormalize",
    "to_tensor",
    # Remote access (Phase 2)
    "open_remote",
    "is_remote_url",
    # Batch migration (Phase 2)
    "migrate_directory",
    "MigrationReport",
    # Streaming (Phase 2)
    "StreamWriter",
    "open_appendable",
    "append_frame",
    # Pipeline adapters (Phase 2)
    "to_ccddata",
    "from_ccddata",
    "to_nddata",
    "nova_to_hdulist",
    # Pipeline framework (Phase 4)
    "Pipeline",
    "Step",
    "PipelineLog",
    "StepLog",
    # Native operations with tracking (Phase 4)
    "OperationHistory",
    "OperationRecord",
    "op_add",
    "op_subtract",
    "op_multiply",
    "op_divide",
    "op_clip",
    "op_mask_replace",
    "op_normalize",
    "op_rebin",
    "op_combine",
    # Math (integrated mathematical tools)
    "math",
    # Visualization
    "viz",
    # Constants
    "constants",
    # Phase 3: Advanced Analysis
    "image_processing",
    "photometry",
    "spectral",
    "coords",
    "catalog",
    # Phase 4: Pipeline tools
    "pipeline",
    "operations",
    "astrometry",
    "photometry_pipeline",
    "spectroscopy_pipeline",
]
