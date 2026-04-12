# NOVA — Next-generation Open Volumetric Archive

**A cloud-native scientific data format for professional astronomy, designed to succeed FITS.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Spec Version](https://img.shields.io/badge/spec-v0.1--draft-orange.svg)](spec/nova-spec-v0.1.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-green.svg)](nova-py/)

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
│   │   └── integrity.py        # SHA-256 chunk integrity
│   ├── tests/
│   └── examples/
│       └── fits_to_nova.py     # Example conversion script
└── README.md
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

## Specification

📄 **[NOVA Format Specification v0.1 (Draft)](spec/nova-spec-v0.1.md)**

## Strategic Roadmap

1. ✅ Solid specification
2. 🔄 Python reference implementation (`nova-py`)
3. ⬜ IVOA endorsement
4. ⬜ Adoption in Rubin LSST / SKA
5. ⬜ Formal ISO standardization

## License

MIT License — See [LICENSE](LICENSE) for details.