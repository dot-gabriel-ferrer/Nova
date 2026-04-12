# NOVA Development Plan — From Prototype to FITS Replacement

**Status:** Active Development  
**Version:** 0.2.0 → 1.0.0 Roadmap  
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

## Current State (v0.2.0)

### ✅ Completed

| Component | Status | Details |
|-----------|--------|---------|
| Format Specification | ✅ v0.2 Draft | 5-layer architecture, 7 design invariants, MEF + tables |
| Container (Zarr v3) | ✅ Complete | Store management, chunking, compression, multi-extension, tables |
| WCS (JSON-LD) | ✅ Complete | 15+ projections, FITS round-trip, per-extension WCS |
| FITS Converter | ✅ Complete | Bidirectional, MEF support, BinTable, scaled integers |
| Provenance (W3C PROV-DM) | ✅ Complete | Entity/Activity/Agent model |
| Integrity (SHA-256) | ✅ Complete | Per-chunk verification |
| Validation | ✅ Complete | JSON Schema, 5 schema files |
| ML Tools | ✅ Complete | 5 normalizations, PyTorch/JAX export |
| Math Tools | ✅ Complete | Statistics, convolution, detection, photometry, spectral |
| Visualization | ✅ Complete | 8 display functions, multiple stretches |
| Benchmarks | ✅ Complete | 5-format comparison, plot generation |
| Fast I/O | ✅ Complete | NOVAFAST binary format, zero-copy |
| CLI | ✅ Complete | convert, info, validate, benchmark |
| Multi-Extension | ✅ **v0.2** | MEF ↔ NOVA, per-extension WCS, auto HDU mapping |
| Table Data | ✅ **v0.2** | Columnar storage, BINTABLE conversion, column metadata |
| Data Types | ✅ **v0.2** | complex64/128, all integer types, scaled integers |
| Test Suite | ✅ 243 tests | All modules + Phase 1 + real image pipeline tests |
| Tutorials | ✅ 6 scripts | Quickstart through math tools |
| Notebooks | ✅ 5 notebooks | Interactive tutorials with visualization |

---

## Phase 1: Core Stability & FITS Parity (v0.2.0) — ✅ COMPLETE

**Goal:** Achieve complete feature parity with FITS, ensuring NOVA can handle every scenario FITS handles.

### 1.1 Multi-Extension Support ✅

- [x] Support reading FITS multi-extension files (MEF) with automatic HDU mapping
- [x] Map FITS extensions → NOVA groups (SCI, ERR, DQ, VARIANCE)
- [x] Support IMAGE, TABLE, and BINTABLE HDU types
- [x] Preserve extension-specific WCS and headers
- [x] Test with synthetic multi-extension FITS files (37 tests)

### 1.2 Table Data Support ✅

- [x] Implement NOVA table storage (columnar arrays in Zarr)
- [x] Support FITS BINTABLE → NOVA table conversion
- [x] Support variable-length arrays (VLA) in tables
- [x] Column metadata with UCDs and units
- [x] Efficient columnar access (read single columns without full table)

### 1.3 Header Keyword Completeness ✅

- [x] Map ALL standard FITS keywords to NOVA JSON-LD properties
- [x] Support HIERARCH keywords (long keyword names)
- [x] Support CONTINUE keywords (long string values)
- [x] Preserve keyword ordering and comments
- [x] Handle non-standard/instrument-specific keywords gracefully

### 1.4 Compression Codecs ✅

- [x] ZSTD compression with configurable levels (default)
- [ ] Add LZ4 compression support (deferred — requires Zarr codec registration)
- [ ] Add JPEG-XL lossy compression for preview images (deferred)
- [x] Configurable per-array compression (different levels per extension)
- [ ] Rice compression compatibility (deferred — low priority)
- [x] Benchmark ZSTD against FITS uncompressed

### 1.5 Data Type Coverage ✅

- [x] Support complex64/complex128 arrays
- [x] Support integer types: int8, uint8, int16, uint16, int32, uint32, int64
- [x] Support scaled integers (BSCALE/BZERO pattern)
- [x] Support unsigned integers via BZERO convention
- [x] float16 support (ML-native, INV-7)

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
| Binary tables | ✅ | ✅ | ✅ Complete (v0.2) |
| ASCII tables | ✅ | ✅ | ✅ Complete (v0.2) |
| Multiple extensions | ✅ | ✅ | ✅ Complete (v0.2) |
| Random groups | ✅ (deprecated) | N/A | Not needed |
| Tile compression | ✅ | ✅ | ✅ (native chunks) |
| Variable-length arrays | ✅ | ✅ | ✅ Complete (v0.2) |
| Complex data types | ✅ | ✅ | ✅ Complete (v0.2) |
| Scaled integers | ✅ | ✅ | ✅ Complete (v0.2) |

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

### Immediate (v0.3.0 — Next Release)

1. **Cloud remote access** — HTTP/S3 store backends
2. **Astropy adapter** — `astropy.io.nova` for seamless integration
3. **Pipeline adapters** — ccdproc, photutils, specutils integration
4. **Batch migration tool** — Convert entire archives
5. **DASK backend** — Parallel chunk processing

### Short-term (v0.4.0)

1. **PSF modelling** — Moffat, multi-Gaussian
2. **Image registration** — WCS-based and feature-based
3. **SIP distortion** — Full WCS distortion support
4. **Advanced photometry** — PSF fitting, crowded fields

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
