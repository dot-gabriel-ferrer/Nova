# NOVA — Next-generation Open Volumetric Archive

**A cloud-native scientific data format for professional astronomy, designed to succeed FITS.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Spec Version](https://img.shields.io/badge/spec-v0.1--draft-orange.svg)](spec/nova-spec-v0.1.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green.svg)](nova-py/)
[![Tests](https://img.shields.io/badge/tests-206%20passed-brightgreen.svg)](nova-py/tests/)

---

## Why NOVA?

FITS (Flexible Image Transport System) has served astronomy for 45+ years, but its
structural limitations are irresolvable:

| Limitation | FITS | NOVA |
|---|---|---|
| Header format | 80-char text cards (IBM punch card origin) | JSON-LD typed metadata |
| Cloud access | Full file download required | Chunk index at byte 0, 2 HTTP requests max |
| Compression | Not in base standard | ZSTD (lossless), LZ4 (speed), JPEG-XL (preview) |
| Metadata validation | Untyped, no schema | JSON Schema + JSON-LD vocabularies |
| Provenance | Not supported | W3C PROV-DM mandatory for reduced data |
| ML compatibility | Manual conversion required | Native float16/BFloat16, zarr→PyTorch/JAX |
| Endianness | Big-endian fixed | Little-endian native (modern hardware) |
| Parallel write | Not supported | Lock-free concurrent writes |
| Math tools | External libraries required | Integrated optimized math operations |
| Visualization | External tools (ds9, etc.) | Built-in display functions |

## Performance: NOVA vs FITS

Benchmarks on realistic astronomical data (Gaussian sky with point sources, float64):

### Overview

![NOVA vs FITS Performance Overview](docs/benchmarks/benchmark_overview.png)

### Cloud Access Advantage

NOVA's chunk-based architecture enables partial reads without downloading the entire file — critical for cloud-based archives:

![Cloud Access Speedup](docs/benchmarks/cloud_access_speedup.png)

### Compression Efficiency

NOVA uses ZSTD lossless compression by default. FITS stores data uncompressed:

![Compression Comparison](docs/benchmarks/compression_comparison.png)

### Improvement Summary

Overall NOVA advantages over FITS at 2048×2048 resolution:

![Improvement Summary](docs/benchmarks/improvement_summary.png)

> **Reproduce these benchmarks:** `nova benchmark --size 2048 --pattern realistic_sky`

## Architecture (5 Layers)

```
┌─────────────────────────────────────────────────────┐
│  Layer 5: ASTRONOMICAL SEMANTICS                    │
│  WCS structured · UCDs IVOA · W3C PROV · Instrument │
├─────────────────────────────────────────────────────┤
│  Layer 4: METADATA                                  │
│  JSON-LD typed · Schema validation · Semver         │
├─────────────────────────────────────────────────────┤
│  Layer 3: SCIENTIFIC DATA                           │
│  N-dim arrays · float16/32/64/BF16 · Complex64/128 │
├─────────────────────────────────────────────────────┤
│  Layer 2: COMPRESSION & ENCODING                    │
│  ZSTD lossless · LZ4 speed · Little-endian native   │
├─────────────────────────────────────────────────────┤
│  Layer 1: CONTAINER & ACCESS                        │
│  Zarr v3 · Chunk index byte 0 · HTTP Range native   │
└─────────────────────────────────────────────────────┘
```

## Design Invariants

1. **BACKWARD_COMPAT** — Lossless FITS↔NOVA conversion. Every existing FITS file must be importable.
2. **CLOUD_FIRST** — Chunk index in the first 8KB. Any region accessible in ≤2 HTTP requests.
3. **HUMAN_READABLE** — JSON-LD metadata manifest readable with any text editor.
4. **INTEGRITY_BY_DEFAULT** — SHA-256 per chunk in the index. Automatic verification on read.
5. **PROV_MANDATORY** — W3C PROV-DM provenance required for reduced data files.
6. **PARALLEL_WRITE** — Concurrent writes from multiple processes/nodes, no global locks.
7. **ML_NATIVE** — Native float16/BFloat16, standardized normalization metadata, zarr→PyTorch/JAX.

## Repository Structure

```
Nova/
├── spec/                        # Format specification
│   ├── nova-spec-v0.1.md       # Full specification document
│   ├── schemas/                 # JSON Schemas
│   │   ├── wcs.schema.json     # WCS JSON-LD schema
│   │   ├── nova-metadata.schema.json
│   │   └── provenance.schema.json
│   └── examples/                # Example JSON-LD documents
│       ├── wcs-tangent-projection.json
│       └── fits-header-converted.json
├── nova-py/                     # Python reference implementation
│   ├── nova/
│   │   ├── container.py        # Zarr v3 container management
│   │   ├── wcs.py              # WCS JSON-LD handling
│   │   ├── fits_converter.py   # FITS↔NOVA converter
│   │   ├── provenance.py       # W3C PROV-DM support
│   │   ├── integrity.py        # SHA-256 chunk integrity
│   │   ├── validation.py       # Schema validation
│   │   ├── ml.py               # ML-native tensor support (INV-7)
│   │   ├── math.py             # Integrated math operations
│   │   ├── visualization.py    # Display & plotting tools
│   │   ├── benchmarks.py       # Performance benchmarking
│   │   ├── plots.py            # Benchmark plot generation
│   │   ├── fast_io.py          # High-performance binary I/O
│   │   └── cli.py              # Command-line interface
│   ├── tests/                   # 206 tests
│   ├── tutorials/               # Step-by-step tutorials
│   │   ├── 01_quickstart.py
│   │   ├── 02_fits_conversion.py
│   │   ├── 03_cloud_access.py
│   │   ├── 04_provenance.py
│   │   └── 05_performance.py
│   └── examples/
│       └── fits_to_nova.py     # Example conversion script
├── notebooks/                   # Jupyter notebooks
│   ├── 01_NOVA_Quickstart.ipynb
│   ├── 02_FITS_to_NOVA_Migration.ipynb
│   ├── 03_Performance_Benchmarks.ipynb
│   ├── 04_Real_Astronomical_Data.ipynb
│   └── 05_Math_and_Visualization_Tools.ipynb
├── docs/
│   └── benchmarks/              # Generated performance plots
│       ├── benchmark_overview.png
│       ├── cloud_access_speedup.png
│       ├── compression_comparison.png
│       └── improvement_summary.png
└── README.md
```

## Installation

```bash
pip install -e nova-py            # Core library
pip install -e "nova-py[plots]"   # + plot generation
pip install -e "nova-py[ml]"      # + PyTorch/JAX support
pip install -e "nova-py[notebooks]" # + Jupyter support
pip install -e "nova-py[all]"     # Everything
```

## Quick Start

```python
import nova

# Convert FITS to NOVA
nova.from_fits("observation.fits", "observation.nova.zarr")

# Read NOVA file
ds = nova.open("observation.nova.zarr")
print(ds.wcs)           # Structured WCS object
print(ds.provenance)    # Full provenance chain
print(ds.data[:100,:100])  # Lazy chunk-based access

# Cloud access (2 requests max)
ds = nova.open("https://archive.example.org/obs/12345.nova.zarr")
cutout = ds.data[1000:1100, 2000:2100]  # Only fetches needed chunks
```

### ML-Native Tensor Export (INV-7)

```python
from nova.ml import to_tensor, compute_normalization, normalize

# Prepare data for PyTorch/JAX
tensor, norm_meta = to_tensor(
    ds.data[:],
    dtype="float32",
    normalize_method="z_score",
    add_batch_dim=True,
    add_channel_dim=True,
)
# tensor shape: (1, 1, H, W), ready for CNNs

# Or use PyTorch directly
from nova.ml import to_pytorch
torch_tensor = to_pytorch(ds.data[:], normalize_method="min_max")
```

### Integrated Math Tools

```python
from nova.math import (
    sigma_clipped_stats, estimate_background,
    detect_sources, aperture_photometry,
    stack_images, smooth_gaussian,
)

# Background estimation and source detection
bg, rms = estimate_background(data, box_size=64)
sources = detect_sources(data - bg, nsigma=5.0)

# Aperture photometry
for src in sources:
    phot = aperture_photometry(
        data, x=src["x"], y=src["y"],
        radius=8.0, annulus_inner=12.0, annulus_outer=18.0,
    )
    print(f"Source at ({src['x']:.0f}, {src['y']:.0f}): flux = {phot['flux_corrected']:.0f}")

# Image stacking (sigma-clipped)
stacked = stack_images(exposures, method="sigma_clip", sigma=3.0)
```

### Easy Visualization

```python
from nova import viz

# Quick-look image with stretch
viz.display_image(data, stretch="asinh", cmap="gray", output_path="preview.png")

# RGB composite
viz.display_rgb(red, green, blue, stretch="asinh")

# Side-by-side comparison
viz.display_comparison(original, processed, show_difference=True)
```

### Validate a NOVA Store

```python
import nova

results = nova.validate("observation.nova.zarr")
for filename, errors in results.items():
    if errors:
        print(f"✗ {filename}: {errors}")
    else:
        print(f"✓ {filename}")
```

## CLI Usage

```bash
# Convert FITS ↔ NOVA
nova convert observation.fits observation.nova.zarr
nova convert observation.nova.zarr output.fits

# Show dataset information
nova info observation.nova.zarr

# Validate against NOVA spec
nova validate observation.nova.zarr

# Run performance benchmarks
nova benchmark --size 2048 --pattern realistic_sky
```

## Tutorials

Step-by-step Python tutorials (runnable scripts):

| # | Tutorial | Description |
|---|---|---|
| 01 | [Quickstart](nova-py/tutorials/01_quickstart.py) | Create your first NOVA dataset from scratch |
| 02 | [FITS Conversion](nova-py/tutorials/02_fits_conversion.py) | FITS↔NOVA migration with round-trip verification |
| 03 | [Cloud Access](nova-py/tutorials/03_cloud_access.py) | Cloud-native chunk-based data retrieval |
| 04 | [Provenance](nova-py/tutorials/04_provenance.py) | W3C PROV-DM data lineage tracking |
| 05 | [Performance](nova-py/tutorials/05_performance.py) | NOVA vs FITS performance benchmarks |

```bash
cd nova-py
python tutorials/01_quickstart.py
```

## Jupyter Notebooks

Interactive notebooks with visualizations and charts:

| # | Notebook | Description |
|---|---|---|
| 01 | [NOVA Quickstart](notebooks/01_NOVA_Quickstart.ipynb) | Interactive tutorial with data visualization |
| 02 | [FITS Migration](notebooks/02_FITS_to_NOVA_Migration.ipynb) | Complete migration guide with metadata inspection |
| 03 | [Performance Benchmarks](notebooks/03_Performance_Benchmarks.ipynb) | Interactive benchmarks with charts |
| 04 | [Real Astronomical Data](notebooks/04_Real_Astronomical_Data.ipynb) | Full pipeline: detection, photometry, stacking |
| 05 | [Math & Visualization](notebooks/05_Math_and_Visualization_Tools.ipynb) | Integrated math and display tools |

```bash
pip install -e "nova-py[notebooks]"
jupyter notebook notebooks/
```

## Specification

📄 **[NOVA Format Specification v0.1 (Draft)](spec/nova-spec-v0.1.md)**

## Implementation Status

| Module | Status | Tests | Description |
|---|---|---|---|
| `container.py` | ✅ Complete | 7 tests | Zarr v3 store management |
| `wcs.py` | ✅ Complete | 14 tests | Structured WCS (JSON-LD) |
| `fits_converter.py` | ✅ Complete | 4 tests | Bidirectional FITS↔NOVA conversion |
| `provenance.py` | ✅ Complete | 8 tests | W3C PROV-DM provenance |
| `integrity.py` | ✅ Complete | 8 tests | SHA-256 chunk verification |
| `validation.py` | ✅ Complete | 16+3 tests | JSON Schema validation |
| `ml.py` | ✅ Complete | 18 tests | ML-native tensor export (INV-7) |
| `math.py` | ✅ **New** | 49 tests | Integrated math operations |
| `visualization.py` | ✅ **New** | 18 tests | Display & plotting tools |
| `benchmarks.py` | ✅ Complete | 18 tests | Performance benchmarking |
| `fast_io.py` | ✅ Complete | 12 tests | High-performance binary I/O |
| `cli.py` | ✅ Complete | 9 tests | Command-line interface |
| `plots.py` | ✅ Complete | — | Benchmark plot generation |
| *Real image tests* | ✅ **New** | 17 tests | Full pipeline with realistic data |

**Total: 206 tests passing**

## Strategic Roadmap

1. ✅ Solid specification (v0.1 draft complete)
2. ✅ Python reference implementation (`nova-py` — all 7 design invariants implemented)
3. ✅ Integrated math & visualization tools
4. ✅ Comprehensive test suite with real astronomical data (206 tests)
5. ⬜ Multi-extension FITS support & table data (v0.2)
6. ⬜ Cloud remote access (S3, HTTP) & pipeline adapters (v0.3)
7. ⬜ Performance optimization & large-scale support (v0.5)
8. ⬜ IVOA endorsement & ecosystem (v0.8)
9. ⬜ Formal standardization (v1.0)

📋 **[Full Development Plan →](DEVELOPMENT_PLAN.md)**

## License

MIT License — See [LICENSE](LICENSE) for details.