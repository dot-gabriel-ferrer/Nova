# NOVA Development Plan -- From Prototype to FITS Replacement

**Status:** Active Development
**Version:** 0.4.0 -> 1.0.0 Roadmap
**Updated:** 2026-04-13

---

## Vision

NOVA (Next-generation Open Volumetric Archive) is designed to be a drop-in replacement
for FITS in every astronomical pipeline -- from raw telescope data to archived science
products. The goal is to provide a format that is:

- **Faster** than FITS (zero-copy I/O, ZSTD compression, little-endian native)
- **Smarter** than FITS (typed JSON-LD metadata, schema validation, provenance)
- **Cloud-native** (chunk-indexed, partial reads, HTTP Range support)
- **ML-ready** (native float16/BFloat16, tensor export, normalization metadata)
- **Self-contained** (integrated math, visualization, and analysis tools)
- **Pipeline-compatible** (FITS<->NOVA round-trip, minimal migration effort)

---

## Current State (v0.4.0)

### Completed

| Component | Version | Details |
|-----------|---------|---------|
| Format Specification | v0.3 Draft | 5-layer architecture, 7 design invariants, MEF + tables |
| Container (Zarr v3) | v0.1 | Store management, chunking, compression, multi-ext, tables |
| WCS (JSON-LD) | v0.1 | 15+ projections, FITS round-trip, per-extension WCS |
| FITS Converter | v0.2 | Bidirectional, MEF support, BinTable, scaled integers |
| Provenance (W3C PROV-DM) | v0.1 | Entity/Activity/Agent model |
| Integrity (SHA-256) | v0.1 | Per-chunk verification |
| Validation | v0.1 | JSON Schema, 5 schema files |
| ML Tools | v0.1 | 5 normalizations, PyTorch/JAX export |
| Math Tools | v0.1 | Statistics, convolution, detection, photometry, spectral |
| Visualization | v0.1 | 8 display functions, multiple stretches |
| Benchmarks | v0.1 | 5-format comparison, plot generation |
| Fast I/O | v0.1 | NOVAFAST binary format, zero-copy |
| CLI | v0.2 | convert, info, validate, benchmark, migrate |
| Multi-Extension | v0.2 | MEF <-> NOVA, per-extension WCS, auto HDU mapping |
| Table Data | v0.2 | Columnar storage, BINTABLE conversion, column metadata |
| Data Types | v0.2 | complex64/128, all integer types, scaled integers |
| Remote Access | v0.3 | HTTP/S3/GCS/Azure via fsspec, lazy chunk loading |
| Batch Migration | v0.3 | Directory conversion, parallel, verify, incremental |
| Streaming | v0.3 | Append-mode writes, time-series ingest, buffered I/O |
| Pipeline Adapters | v0.3 | CCDData, NDData, HDUList bidirectional adapters |
| Image Processing | v0.4 | PSF modelling, registration, subtraction, calibration |
| Advanced Photometry | v0.4 | PSF fitting, extended source, calibration, crowded-field |
| Advanced Spectral | v0.4 | Wavelength calib, sky subtraction, line fitting, echelle |
| Coordinate Transforms | v0.4 | SIP, TPV, lookup distortion, frame transforms, cross-match |
| Catalog Operations | v0.4 | Cross-matching, cone search, VOTable, SAMP |
| Test Suite | v0.4 | 370 tests across all modules and features |
| Constants | v0.3 | Centralised shared values (MAD_TO_STD, HASH_READ_SIZE, etc.) |
| Tutorials | v0.3 | 7 scripts covering all features |
| Notebooks | v0.1 | 5 interactive notebooks with visualization |

---

## Phase 1: Core Stability and FITS Parity (v0.2.0) -- COMPLETE

**Goal:** Achieve complete feature parity with FITS, ensuring NOVA can handle
every scenario FITS handles.

### 1.1 Multi-Extension Support [done]

- [x] Support reading FITS multi-extension files (MEF) with automatic HDU mapping
- [x] Map FITS extensions -> NOVA groups (SCI, ERR, DQ, VARIANCE)
- [x] Support IMAGE, TABLE, and BINTABLE HDU types
- [x] Preserve extension-specific WCS and headers
- [x] Test with synthetic multi-extension FITS files (37 tests)

### 1.2 Table Data Support [done]

- [x] Implement NOVA table storage (columnar arrays in Zarr)
- [x] Support FITS BINTABLE -> NOVA table conversion
- [x] Support variable-length arrays (VLA) in tables
- [x] Column metadata with UCDs and units
- [x] Efficient columnar access (read single columns without full table)

### 1.3 Header Keyword Completeness [done]

- [x] Map ALL standard FITS keywords to NOVA JSON-LD properties
- [x] Support HIERARCH keywords (long keyword names)
- [x] Support CONTINUE keywords (long string values)
- [x] Preserve keyword ordering and comments
- [x] Handle non-standard/instrument-specific keywords gracefully

### 1.4 Compression Codecs [done]

- [x] ZSTD compression with configurable levels (default)
- [ ] Add LZ4 compression support (deferred -- requires Zarr codec registration)
- [ ] Add JPEG-XL lossy compression for preview images (deferred)
- [x] Configurable per-array compression (different levels per extension)
- [ ] Rice compression compatibility (deferred -- low priority)
- [x] Benchmark ZSTD against FITS uncompressed

### 1.5 Data Type Coverage [done]

- [x] Support complex64/complex128 arrays
- [x] Support integer types: int8, uint8, int16, uint16, int32, uint32, int64
- [x] Support scaled integers (BSCALE/BZERO pattern)
- [x] Support unsigned integers via BZERO convention
- [x] float16 support (ML-native, INV-7)

---

## Phase 2: Cloud and Pipeline Integration (v0.3.0) -- COMPLETE

**Goal:** Make NOVA work seamlessly in cloud-based astronomical pipelines.

### 2.1 Remote Store Access [done]

- [x] Support opening NOVA stores from HTTP/HTTPS URLs
- [x] Support opening from S3 (boto3 backend via fsspec)
- [x] Support opening from Google Cloud Storage (gcsfs)
- [x] Support opening from Azure Blob Storage (adlfs)
- [x] Lazy loading: only fetch chunks on access
- [x] Authenticated access via storage_options (OAuth2, AWS STS tokens)

### 2.2 Pipeline Adapters [done]

- [x] CCDData adapter -- convert NOVA <-> astropy CCDData
- [x] NDData adapter -- lightweight astropy NDData conversion
- [x] HDUList adapter -- convert NOVA -> in-memory astropy HDUList
- [x] Bidirectional round-trip (from_ccddata, to_ccddata)
- [ ] Full ccdproc compatibility layer (deferred -- needs ccdproc testing)
- [ ] photutils integration (deferred)
- [ ] specutils integration (deferred)
- [ ] DASK array backend (deferred to Phase 4)

### 2.3 Pipeline Migration Tool [done]

- [x] Automated FITS->NOVA migration script for entire directories/archives
- [x] Parallel batch conversion (multiprocessing)
- [x] Verification report after migration (checksums, data comparison)
- [x] Dry-run mode (analyze without converting)
- [x] Incremental migration (skip already converted files)
- [x] CLI: `nova migrate` subcommand

### 2.4 Streaming I/O [done]

- [x] Support appending data to existing arrays (time series, monitoring)
- [x] Support streaming writes (data arrives in chunks)
- [x] Buffered StreamWriter with configurable flush interval
- [x] Metadata written automatically on close
- [ ] WebSocket support for live data streaming (deferred)
- [ ] Real-time telescope control system ingestion (deferred)

---

## Post-Development Audit (v0.3.1) -- COMPLETE

**Goal:** Improve code quality, eliminate magic numbers, centralise shared
constants, add input validation to all public API entry points, and harden
error handling.

### A.1 Constants Centralisation [done]

- [x] Create nova/constants.py with all shared values
- [x] Migrate NOVA_VERSION, NOVA_CONTEXT from container.py, wcs.py, provenance.py, streaming.py
- [x] Replace magic number 1.4826 with MAD_TO_STD in ml.py and math.py
- [x] Replace magic number 65536 with HASH_READ_SIZE in integrity.py
- [x] Replace magic number 65536 with TABLE_CHUNK_SIZE in container.py
- [x] Replace hardcoded 3600 with ARCSEC_PER_DEG in wcs.py
- [x] Centralise FITS_EXTENSIONS in constants.py (used by migrate.py)
- [x] Centralise SUPPORTED_REMOTE_SCHEMES in constants.py (used by remote.py)
- [x] Make SUPPORTED_DTYPES, SUPPORTED_CODECS, FITS_EXTENSIONS frozensets (immutable)
- [x] Consolidate hardcoded validation URLs with NOVA_CONTEXT in validation.py

### A.2 Input Validation [done]

- [x] container.set_science_data: reject non-ndarray inputs (TypeError)
- [x] container.set_uncertainty: reject non-ndarray inputs (TypeError)
- [x] container.set_mask: reject non-ndarray inputs (TypeError)
- [x] remote.RemoteNovaDataset: validate URL scheme on construction
- [x] streaming.StreamWriter: validate buffer_size >= 1
- [x] streaming.StreamWriter: validate compression_level 0-22

### A.3 Error Handling [done]

- [x] Replace bare Exception catch in migrate._convert_single_file with specific types
- [x] Keep narrow exception set in _verify_roundtrip (OSError, ValueError, TypeError, KeyError)

### A.4 Audit Test Suite [done]

- [x] 22 new tests in test_audit.py covering all audit improvements
- [x] Tests verify constants are frozenset where expected
- [x] Tests verify modules import from constants (no hardcoded duplicates)
- [x] Tests verify TypeError raised for invalid input types
- [x] Tests verify ValueError raised for invalid URL schemes
- [x] Total test suite: 292 tests passing

---

## Phase 3: Advanced Math and Analysis (v0.4.0) -- COMPLETE

**Goal:** Make NOVA a complete analysis environment, reducing the need for external tools.

### 3.1 Advanced Image Processing [done]

- [x] PSF modelling (Moffat, multi-Gaussian)
- [x] Image registration/alignment (WCS-based and feature-based)
- [x] Image subtraction (Alard-Lupton method)
- [x] Flat-fielding and bias subtraction helpers
- [x] Bad pixel interpolation
- [x] Fringe correction for CCD images

### 3.2 Advanced Photometry [done]

- [x] PSF photometry (simultaneous PSF fitting)
- [x] Extended source photometry (Petrosian, Kron, isophotal)
- [x] Photometric calibration helpers (zero-point, extinction)
- [x] Aperture corrections
- [x] Crowded-field photometry

### 3.3 Advanced Spectral Tools [done]

- [x] Wavelength calibration (arc lamp fitting)
- [x] Sky subtraction for spectra
- [x] Radial velocity measurement
- [x] Spectral classification helpers
- [x] Emission line fitting (Gaussian, Voigt profiles)
- [x] Multi-order echelle spectrum support

### 3.4 Coordinate Transforms [done]

- [x] Full SIP distortion support
- [x] TPV (tangent polynomial) distortion
- [x] Lookup table (IMAGE-type) distortion
- [x] Coordinate transformation between WCS frames
- [x] Astrometric solution verification (catalog cross-match)

### 3.5 Catalog Operations [done]

- [x] Cross-matching between NOVA catalogs
- [x] Cone search (spatial queries)
- [x] VO Table (VOTable) import/export
- [x] SAMP integration (interoperability with other tools)

---

## Phase 4: Performance and Scale (v0.5.0)

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

## Phase 5: Ecosystem and Adoption (v0.8.0)

**Goal:** Build the ecosystem needed for widespread adoption.

### 5.1 Language Bindings

- [ ] C library (libnova) -- for integration with compiled pipelines
- [ ] Julia package (Nova.jl) -- for the growing Julia astronomy community
- [ ] Rust library (nova-rs) -- for high-performance services
- [ ] JavaScript/TypeScript (nova-js) -- for web-based visualisation

### 5.2 Viewer Applications

- [ ] nova-viewer -- lightweight desktop viewer (Qt/Tkinter)
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

### 5.4 Documentation and Community

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
| 2D images | Yes | Yes | Complete |
| 3D data cubes | Yes | Yes | Complete |
| N-D arrays | Yes | Yes | Complete |
| Binary tables | Yes | Yes | Complete (v0.2) |
| ASCII tables | Yes | Yes | Complete (v0.2) |
| Multiple extensions | Yes | Yes | Complete (v0.2) |
| Random groups | Yes (deprecated) | N/A | Not needed |
| Tile compression | Yes | Yes | Complete (native chunks) |
| Variable-length arrays | Yes | Yes | Complete (v0.2) |
| Complex data types | Yes | Yes | Complete (v0.2) |
| Scaled integers | Yes | Yes | Complete (v0.2) |

### Metadata Features

| Feature | FITS | NOVA | Status |
|---------|------|------|--------|
| Header keywords | Yes (80-char) | Yes (JSON-LD) | Complete |
| WCS | Yes | Yes (structured) | Complete |
| Keyword types | Limited | Full JSON types | Complete |
| Schema validation | No | Yes | Complete |
| Provenance | No | Yes (W3C PROV-DM) | Complete |
| HIERARCH keys | Yes | Yes (native) | Complete |
| HISTORY/COMMENT | Yes | Yes (JSON arrays) | Complete |

### Tools Comparison

| Tool | FITS Equivalent | NOVA | Status |
|------|----------------|------|--------|
| fitsinfo/fitsheader | nova info | CLI | Complete |
| fitsverify | nova validate | CLI | Complete |
| fpack/funpack | Native compression | Built-in | Complete |
| ds9/JS9 viewer | nova.viz | Module | Complete |
| IRAF imstat | nova.math stats | Module | Complete |
| SExtractor detect | nova.math detect | Module | Complete |
| photutils aperture | nova.math aperture | Module | Complete |
| swarp stack | nova.math stack | Module | Complete |

### Pipeline Integration (v0.3)

| Feature | Status | Details |
|---------|--------|---------|
| Remote HTTP access | Complete | fsspec + lazy Zarr |
| S3 cloud storage | Complete | boto3/s3fs backend |
| GCS cloud storage | Complete | gcsfs backend |
| Azure Blob storage | Complete | adlfs backend |
| astropy CCDData | Complete | Bidirectional adapter |
| astropy NDData | Complete | Lightweight adapter |
| Batch migration | Complete | Parallel, verify, incremental |
| Time-series append | Complete | StreamWriter with buffering |

### Advanced Analysis (v0.4)

| Feature | Status | Details |
|---------|--------|---------|
| PSF modelling | Complete | Moffat, multi-Gaussian, fitting |
| Image registration | Complete | WCS-based and feature-based alignment |
| Image subtraction | Complete | Scaled and Alard-Lupton methods |
| CCD calibration | Complete | Bias, flat, dark, overscan, fringe |
| Bad pixel handling | Complete | Interpolation, mask building |
| PSF photometry | Complete | Simultaneous fitting, iterative subtraction |
| Extended photometry | Complete | Petrosian, Kron, isophotal, radial profile |
| Photometric calibration | Complete | Zero-point, extinction, color terms |
| Crowded-field photometry | Complete | Neighbor finding, deblending, completeness |
| Wavelength calibration | Complete | Arc lamp fitting, polynomial solution |
| Sky subtraction | Complete | 1-D and 2-D methods |
| Radial velocity | Complete | Cross-correlation, Doppler shift |
| Emission line fitting | Complete | Gaussian, Voigt, multi-line |
| Echelle support | Complete | Order tracing, extraction, merging |
| SIP distortion | Complete | Forward, inverse, fitting |
| TPV distortion | Complete | Full polynomial evaluation |
| Lookup distortion | Complete | IMAGE-type correction tables |
| Frame transforms | Complete | Galactic, ecliptic, precession |
| Catalog cross-match | Complete | Nearest-neighbor, cone, box, polygon search |
| VOTable support | Complete | Read, write, convert to/from NOVA |
| SAMP integration | Complete | Client for tool interoperability |

---

## Pipeline Migration Guide

### Step 1: Assessment (No code changes)

```bash
# Analyze existing FITS files
nova info *.fits

# Check conversion compatibility
nova migrate raw_fits/ nova_archive/ --dry-run
```

### Step 2: Convert (Parallel, reversible)

```bash
# Batch convert FITS -> NOVA
nova migrate raw_fits/ nova_archive/ --parallel 4 --verify

# Or convert individual files
nova convert observation.fits observation.nova.zarr
```

### Step 3: Update Pipeline Code

```python
# Before (FITS)
from astropy.io import fits
hdul = fits.open("observation.fits")
data = hdul[0].data
header = hdul[0].header

# After (NOVA) -- minimal changes
import nova
ds = nova.open("observation.nova.zarr")
data = ds.data[:]           # Same NumPy array
header = ds.metadata        # JSON dict instead of FITS Header
wcs = ds.wcs                # Structured WCS object

# Or use the CCDData adapter for zero-effort migration
from nova.adapters import to_ccddata
ccd = to_ccddata("observation.nova.zarr", unit="adu")
# ccd works exactly like CCDData from astropy
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

# Time-series streaming (impossible with FITS)
from nova.streaming import open_appendable, append_frame
writer = open_appendable("timeseries.nova.zarr", frame_shape=(256, 256))
for frame in camera_stream():
    append_frame(writer, frame)
writer.close()
```

---

## Development Priorities (Next Steps)

### Immediate (v0.5.0 -- Next Release)

1. **Performance optimization** -- Cython/Numba acceleration
2. **DASK backend** -- Parallel chunk processing
3. **Large-scale support** -- >1 TB datasets
4. **GPU acceleration** -- CuPy optional backend

### Short-term (v0.8.0)

1. **Language bindings** -- C, Julia, Rust
2. **Viewer applications** -- Desktop and web
3. **Archive integration** -- IVOA TAP, SODA

### Long-term (v1.0.0)

1. **IVOA standardization** -- Formal recommendation
2. **Survey pipeline integration** -- Rubin LSST, SKA
3. **ISO standardization** -- International standard
4. **Full ecosystem** -- Viewers, editors, archive services

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
