# NOVA Development Plan — From Prototype to FITS Replacement

**Status:** Active Development  
**Version:** 0.1.0 → 1.0.0 Roadmap  
**Updated:** 2026-04-12

---

## Vision

NOVA (Next-generation Open Volumetric Archive) is designed to be a **drop-in replacement for FITS** in every astronomical pipeline — from raw telescope data to archived science products. The goal is to provide a format that is:

- **Faster** than FITS (zero-copy I/O, ZSTD compression, little-endian native)
- **Smarter** than FITS (typed JSON-LD metadata, schema validation, provenance)
- **Cloud-native** (chunk-indexed, partial reads, HTTP Range support)
- **ML-ready** (native float16/BFloat16, tensor export, normalization metadata)
- **Self-contained** (integrated math, visualization, and analysis tools)
- **Pipeline-compatible** (FITS↔NOVA round-trip, minimal migration effort)

---

## Current State (v0.1.0)

### ✅ Completed

| Component | Status | Details |
|-----------|--------|---------|
| Format Specification | ✅ v0.1 Draft | 5-layer architecture, 7 design invariants |
| Container (Zarr v3) | ✅ Complete | Store management, chunking, compression |
| WCS (JSON-LD) | ✅ Complete | 15+ projections, FITS round-trip |
| FITS Converter | ✅ Complete | Bidirectional, lossless round-trip |
| Provenance (W3C PROV-DM) | ✅ Complete | Entity/Activity/Agent model |
| Integrity (SHA-256) | ✅ Complete | Per-chunk verification |
| Validation | ✅ Complete | JSON Schema, 3 schema files |
| ML Tools | ✅ Complete | 5 normalizations, PyTorch/JAX export |
| Math Tools | ✅ Complete | Statistics, convolution, detection, photometry, spectral |
| Visualization | ✅ Complete | 8 display functions, multiple stretches |
| Benchmarks | ✅ Complete | 5-format comparison, plot generation |
| Fast I/O | ✅ Complete | NOVAFAST binary format, zero-copy |
| CLI | ✅ Complete | convert, info, validate, benchmark |
| Test Suite | ✅ 206 tests | All modules + real image pipeline tests |
| Tutorials | ✅ 5 scripts | Quickstart through performance |
| Notebooks | ✅ 5 notebooks | Interactive tutorials with visualization |

---

## Phase 1: Core Stability & FITS Parity (v0.2.0)

**Goal:** Achieve complete feature parity with FITS, ensuring NOVA can handle every scenario FITS handles.

### 1.1 Multi-Extension Support

- [ ] Support reading FITS multi-extension files (MEF) with automatic HDU mapping
- [ ] Map FITS extensions → NOVA groups (SCI, ERR, DQ, VARIANCE)
- [ ] Support IMAGE, TABLE, and BINTABLE HDU types
- [ ] Preserve extension-specific WCS and headers
- [ ] Test with HST/JWST multi-extension FITS files

### 1.2 Table Data Support

- [ ] Implement NOVA table storage (structured arrays in Zarr)
- [ ] Support FITS BINTABLE → NOVA table conversion
- [ ] Support variable-length arrays (VLA) in tables
- [ ] Column metadata with UCDs and units
- [ ] Efficient columnar access (read single columns without full table)

### 1.3 Header Keyword Completeness

- [ ] Map ALL standard FITS keywords to NOVA JSON-LD properties
- [ ] Support HIERARCH keywords (long keyword names)
- [ ] Support CONTINUE keywords (long string values)
- [ ] Preserve keyword ordering and comments
- [ ] Handle non-standard/instrument-specific keywords gracefully

### 1.4 Compression Codecs

- [ ] Add LZ4 compression support
- [ ] Add JPEG-XL lossy compression for preview images
- [ ] Configurable per-array compression (different codecs per extension)
- [ ] Rice compression compatibility (for integer data like FITS tiled)
- [ ] Benchmark all codecs against FITS cfitsio compression

### 1.5 Data Type Coverage

- [ ] Support complex64/complex128 arrays
- [ ] Support integer types: int8, uint8, int16, uint16, int32, uint32, int64
- [ ] Support scaled integers (BSCALE/BZERO pattern)
- [ ] Support unsigned integers via BZERO convention
- [ ] Bit array support for mask data

---

## Phase 2: Cloud & Pipeline Integration (v0.3.0)

**Goal:** Make NOVA work seamlessly in cloud-based astronomical pipelines.

### 2.1 Remote Store Access

- [ ] Support opening NOVA stores from HTTP/HTTPS URLs
- [ ] Support opening from S3 (boto3 backend)
- [ ] Support opening from Google Cloud Storage
- [ ] Support opening from Azure Blob Storage
- [ ] Lazy loading: only fetch chunks on access
- [ ] Authenticated access (OAuth2, AWS STS tokens)

### 2.2 Pipeline Adapters

- [ ] `astropy.io.nova` adapter — read/write NOVA like FITS via astropy
- [ ] `ccdproc` compatibility layer — use NOVA as drop-in for CCDData
- [ ] `photutils` integration — read NOVA catalogs, write photometry results
- [ ] `specutils` integration — read/write NOVA spectra
- [ ] `reproject` integration — reproject NOVA images to different WCS
- [ ] DASK array backend — parallel processing of NOVA chunks
- [ ] Xarray integration — labeled N-D arrays from NOVA stores

### 2.3 Pipeline Migration Tool

- [ ] Automated FITS→NOVA migration script for entire directories/archives
- [ ] Parallel batch conversion (multiprocessing)
- [ ] Verification report after migration (checksums, WCS, metadata)
- [ ] Dry-run mode (analyze without converting)
- [ ] Incremental migration (skip already converted files)
- [ ] Rollback capability (keep FITS originals, symlink)

### 2.4 Streaming I/O

- [ ] Support appending data to existing arrays (time series, monitoring)
- [ ] Support streaming writes (data arrives in chunks)
- [ ] Real-time data ingestion from telescope control systems
- [ ] WebSocket support for live data streaming

---

## Phase 3: Advanced Math & Analysis (v0.4.0)

**Goal:** Make NOVA a complete analysis environment, reducing the need for external tools.

### 3.1 Advanced Image Processing

- [ ] PSF modelling (Moffat, multi-Gaussian)
- [ ] Image registration/alignment (WCS-based and feature-based)
- [ ] Image subtraction (Alard-Lupton method)
- [ ] Flat-fielding and bias subtraction helpers
- [ ] Bad pixel interpolation
- [ ] Fringe correction for CCD images

### 3.2 Advanced Photometry

- [ ] PSF photometry (simultaneous PSF fitting)
- [ ] Extended source photometry (Petrosian, Kron, isophotal)
- [ ] Photometric calibration helpers (zero-point, extinction)
- [ ] Aperture corrections
- [ ] Crowded-field photometry

### 3.3 Advanced Spectral Tools

- [ ] Wavelength calibration (arc lamp fitting)
- [ ] Sky subtraction for spectra
- [ ] Radial velocity measurement
- [ ] Spectral classification helpers
- [ ] Emission line fitting (Gaussian, Voigt profiles)
- [ ] Multi-order echelle spectrum support

### 3.4 Coordinate Transforms

- [ ] Full SIP distortion support
- [ ] TPV (tangent polynomial) distortion
- [ ] Lookup table (IMAGE-type) distortion
- [ ] Coordinate transformation between WCS frames
- [ ] Astrometric solution verification (catalog cross-match)

### 3.5 Catalog Operations

- [ ] Cross-matching between NOVA catalogs
- [ ] Cone search (spatial queries)
- [ ] VO Table (VOTable) import/export
- [ ] SAMP integration (interoperability with other tools)

---

## Phase 4: Performance & Scale (v0.5.0)

**Goal:** Optimise for production-scale astronomical pipelines.

### 4.1 Performance Optimization

- [ ] Cython/Numba acceleration for critical math operations
- [ ] GPU-accelerated convolution and source detection (CuPy optional)
- [ ] Memory-mapped reading for very large files (>100 GB)
- [ ] Multi-threaded chunk processing
- [ ] Write-ahead logging for crash recovery
- [ ] Benchmark suite with real survey data (SDSS, DES, LSST simulations)

### 4.2 Large-Scale Data

- [ ] Support for >1 TB datasets
- [ ] Hierarchical chunk indexing for very large stores
- [ ] Tiled mosaic support (multi-pointing observations)
- [ ] Time-series cube support (data cubes with time axis)
- [ ] Radio data cubes (RA, Dec, Freq, Stokes)

### 4.3 Parallel Processing

- [ ] DASK integration for parallel array operations
- [ ] MPI-based parallel I/O for HPC clusters
- [ ] Slurm job templates for batch processing
- [ ] Distributed stacking across multiple nodes

---

## Phase 5: Ecosystem & Adoption (v0.8.0)

**Goal:** Build the ecosystem needed for widespread adoption.

### 5.1 Language Bindings

- [ ] C library (`libnova`) — for integration with compiled pipelines
- [ ] Julia package (`Nova.jl`) — for the growing Julia astronomy community
- [ ] Rust library (`nova-rs`) — for high-performance services
- [ ] JavaScript/TypeScript (`nova-js`) — for web-based visualisation

### 5.2 Viewer Applications

- [ ] `nova-viewer` — lightweight desktop viewer (Qt/Tkinter)
- [ ] Web-based viewer (JS, similar to JS9/DS9)
- [ ] VS Code extension for NOVA file inspection
- [ ] Jupyter widget for interactive NOVA display
- [ ] FITS-to-NOVA online converter service

### 5.3 Archive Integration

- [ ] IVOA TAP service compatibility
- [ ] IVOA SODA (Server-side Operations for Data Access) support
- [ ] ObsTAP metadata mapping
- [ ] DOI integration for archived datasets
- [ ] Datalink integration

### 5.4 Documentation & Community

- [ ] API reference documentation (Sphinx/ReadTheDocs)
- [ ] User guide with complete examples
- [ ] Migration guide: "Moving from FITS to NOVA"
- [ ] Contributing guide and code of conduct
- [ ] Example pipelines for common instruments (HST, JWST, VLT, Gemini)
- [ ] Conference presentations and papers
- [ ] Community forum or discussion platform

---

## Phase 6: Standardization (v1.0.0)

**Goal:** Formal recognition and standardization.

### 6.1 IVOA Process

- [ ] Submit NOVA specification as IVOA Note
- [ ] Gather community feedback (12 months)
- [ ] Address all comments and revise specification
- [ ] Submit as IVOA Working Draft
- [ ] Implement reference interoperability tests
- [ ] Submit as IVOA Proposed Recommendation
- [ ] Final IVOA Recommendation

### 6.2 Adoption Milestones

- [ ] 3+ observatories using NOVA in production pipelines
- [ ] Integration in at least one major survey pipeline (Rubin LSST, SKA, Euclid)
- [ ] Support in major astronomy libraries (astropy, photutils, specutils)
- [ ] 1000+ NOVA datasets in public archives
- [ ] Performance whitepaper with independent benchmarks

### 6.3 ISO Standardization (long-term)

- [ ] ISO TC20/SC14 (Space systems) liaison
- [ ] Submit as ISO standard proposal
- [ ] ISO committee review and balloting

---

## FITS Feature Comparison Matrix

This matrix tracks feature parity with FITS and its ecosystem tools:

### Data Features

| Feature | FITS | NOVA | Status |
|---------|------|------|--------|
| 2D images | ✅ | ✅ | ✅ Complete |
| 3D data cubes | ✅ | ✅ | ✅ Complete |
| N-D arrays | ✅ | ✅ | ✅ Complete |
| Binary tables | ✅ | ⬜ | Phase 1 |
| ASCII tables | ✅ | ⬜ | Phase 1 |
| Multiple extensions | ✅ | ⬜ | Phase 1 |
| Random groups | ✅ (deprecated) | N/A | Not needed |
| Tile compression | ✅ | ✅ | ✅ (native chunks) |
| Variable-length arrays | ✅ | ⬜ | Phase 1 |

### Metadata Features

| Feature | FITS | NOVA | Status |
|---------|------|------|--------|
| Header keywords | ✅ (80-char) | ✅ (JSON-LD) | ✅ Complete |
| WCS | ✅ | ✅ (structured) | ✅ Complete |
| Keyword types | Limited | Full JSON types | ✅ Complete |
| Schema validation | ❌ | ✅ | ✅ Complete |
| Provenance | ❌ | ✅ (W3C PROV-DM) | ✅ Complete |
| HIERARCH keys | ✅ | ✅ (native) | ✅ Complete |
| HISTORY/COMMENT | ✅ | ✅ (JSON arrays) | ✅ Complete |

### Tools Comparison

| Tool | FITS Equivalent | NOVA | Status |
|------|----------------|------|--------|
| fitsinfo/fitsheader | nova info | ✅ CLI | ✅ Complete |
| fitsverify | nova validate | ✅ CLI | ✅ Complete |
| fpack/funpack | Native compression | ✅ Built-in | ✅ Complete |
| ds9/JS9 viewer | nova.viz | ✅ Module | ✅ Complete |
| IRAF imstat | nova.math stats | ✅ Module | ✅ Complete |
| SExtractor detect | nova.math detect | ✅ Module | ✅ Complete |
| photutils aperture | nova.math aperture | ✅ Module | ✅ Complete |
| swarp stack | nova.math stack | ✅ Module | ✅ Complete |

---

## Pipeline Migration Guide

### Step 1: Assessment (No code changes)

```bash
# Analyze existing FITS files
nova info *.fits

# Check conversion compatibility
nova convert --dry-run *.fits
```

### Step 2: Convert (Parallel, reversible)

```bash
# Batch convert FITS → NOVA
nova convert *.fits --output-dir nova_data/

# Verify round-trip fidelity
nova convert nova_data/*.nova.zarr --output-dir fits_verify/
diff fits_verify/ original/
```

### Step 3: Update Pipeline Code

```python
# Before (FITS)
from astropy.io import fits
hdul = fits.open("observation.fits")
data = hdul[0].data
header = hdul[0].header

# After (NOVA) — minimal changes
import nova
ds = nova.open("observation.nova.zarr")
data = ds.data[:]           # Same NumPy array
header = ds.metadata        # JSON dict instead of FITS Header
wcs = ds.wcs                # Structured WCS object
```

### Step 4: Leverage NOVA Advantages

```python
# Cloud-native partial reads (was impossible with FITS)
cutout = ds.data[1000:1100, 2000:2100]

# Integrated math (no external dependencies needed)
from nova.math import estimate_background, detect_sources
bg, rms = estimate_background(data, box_size=64)
sources = detect_sources(data - bg, nsigma=5.0)

# ML-ready export (was manual with FITS)
from nova.ml import to_tensor
tensor, meta = to_tensor(data, normalize_method="z_score",
                         add_batch_dim=True, add_channel_dim=True)
```

---

## Development Priorities (Next Steps)

### Immediate (v0.2.0 — Next Release)

1. **Multi-extension FITS support** — Critical for HST/JWST data
2. **Table data support** — Required for source catalogs
3. **LZ4 compression** — Fast codec for real-time pipelines
4. **Cloud remote access** — HTTP/S3 store backends
5. **Astropy adapter** — `astropy.io.nova` for seamless integration

### Short-term (v0.3.0)

1. **Pipeline adapters** — ccdproc, photutils, specutils integration
2. **Batch migration tool** — Convert entire archives
3. **DASK backend** — Parallel chunk processing
4. **SIP distortion** — Full WCS distortion support

### Medium-term (v0.5.0)

1. **Performance optimization** — Cython/Numba acceleration
2. **Large-scale support** — >1 TB datasets
3. **Advanced photometry** — PSF fitting, crowded fields
4. **Language bindings** — C, Julia, Rust

### Long-term (v1.0.0)

1. **IVOA standardization** — Formal recommendation
2. **Survey pipeline integration** — Rubin LSST, SKA
3. **ISO standardization** — International standard
4. **Full ecosystem** — Viewers, editors, archive services

---

## Contributing

We welcome contributions in all areas:

- **Code:** New features, optimizations, bug fixes
- **Tests:** More test data, edge cases, real-world validation
- **Documentation:** Tutorials, examples, API docs
- **Specification:** Review and improve the NOVA spec
- **Adoption:** Try NOVA in your pipeline and report your experience

See the main README for installation and development setup instructions.

---

*This plan is a living document and will be updated as development progresses.*
