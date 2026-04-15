# NOVA Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] -- 2026-04-15

### Added
- Specification updated to v1.0 covering all implemented modules
- CHANGELOG.md with full release history
- CONTRIBUTING.md with development setup and guidelines
- CODE_OF_CONDUCT.md (Contributor Covenant 2.1)
- GitHub Actions CI/CD pipeline (.github/workflows/tests.yml)
- Tutorials 08-12 covering pipeline, operations, astrometry, photometry, spectroscopy
- py.typed marker file for PEP 561 compliance
- pytest-cov configuration in pyproject.toml
- Full type hints on provenance.py and streaming.py

### Changed
- Version bump from 0.5.0 to 1.0.0
- Development Status classifier updated to "4 - Beta"
- Spec updated from v0.3 to v1.0
- README updated for v1.0 release
- DEVELOPMENT_PLAN.md updated with completed audit

---

## [0.5.0] -- 2026-04-13

### Added
- Pipeline framework (pipeline.py) -- declarative pipelines with step logging,
  checksums, and JSON replay
- Native operations with change tracking (operations.py) -- tracked arithmetic
  (add, subtract, multiply, divide), clipping, masking, normalization, rebinning,
  and combine with full JSON serialization
- Astrometry pipeline (astrometry.py) -- centroid extraction with background
  estimation, triangle-matching plate solver, gnomonic projection, proper-motion
  and parallax correction, astrometric residual analysis, SIP fitting
- Photometry pipeline (photometry_pipeline.py) -- multi-aperture photometry with
  Poisson + readnoise errors, zero-point calibration, extinction correction,
  color terms, differential photometry, magnitude conversions, limiting magnitude,
  growth curve analysis
- Spectroscopy pipeline (spectroscopy_pipeline.py) -- optimal extraction (Horne
  1986), continuum fitting, telluric correction, spectral stacking, redshift
  measurement, signal-to-noise estimation, equivalent width, spectral smoothing
- 84 new tests for Phase 4a modules (454 total)

### Changed
- README updated with Phase 4 modules in implementation status table
- pyproject.toml updated with project URLs and Alpha classifier

---

## [0.4.0] -- 2026-04-13

### Added
- Advanced image processing (image_processing.py) -- PSF modelling (Moffat,
  multi-Gaussian), image registration/alignment (WCS and feature-based), image
  subtraction (Alard-Lupton), CCD calibration (bias, flat, dark, overscan,
  fringe), bad pixel interpolation
- Advanced photometry (photometry.py) -- PSF photometry (simultaneous fitting,
  iterative subtraction), extended source photometry (Petrosian, Kron, isophotal,
  radial profile), photometric calibration (zero-point, extinction, color terms),
  crowded-field photometry with deblending and completeness
- Advanced spectral tools (spectral.py) -- wavelength calibration (arc lamp
  fitting), sky subtraction (1-D and 2-D), radial velocity (cross-correlation),
  emission line fitting (Gaussian, Voigt), echelle support (order tracing,
  extraction, merging)
- Coordinate transforms (coords.py) -- SIP distortion (forward, inverse,
  fitting), TPV distortion, lookup (IMAGE-type) distortion, frame transforms
  (galactic, ecliptic, precession), catalog cross-matching
- Catalog operations (catalog.py) -- nearest-neighbor and cone search, box and
  polygon queries, VOTable import/export, SAMP integration
- 66 new tests for Phase 3 modules (370 total)

---

## [0.3.0] -- 2026-04-12

### Added
- Remote store access (remote.py) -- HTTP, S3, GCS, Azure via fsspec with
  lazy chunk loading and authenticated access
- Batch migration tool (migrate.py) -- directory conversion with parallel
  workers, verification, dry-run, and incremental modes
- Streaming I/O (streaming.py) -- append-mode time-series ingest with
  buffered StreamWriter
- Pipeline adapters (adapters.py) -- bidirectional CCDData, NDData, and
  HDUList adapters for astropy compatibility
- Constants module (constants.py) -- centralised shared values replacing
  all magic numbers across the codebase
- CLI migrate subcommand
- Input validation on all public API entry points
- 7 tutorials covering quickstart through migration/streaming
- 5 interactive Jupyter notebooks with visualizations
- Specification v0.3 with remote access, migration, streaming, adapters

### Changed
- Error handling hardened (no bare except clauses)
- All magic numbers replaced with named constants

---

## [0.2.0] -- 2026-04-12

### Added
- Multi-extension FITS support (NovaExtension) with automatic HDU mapping
- Table data support (NovaTable) with columnar storage and BINTABLE conversion
- Variable-length arrays (VLA) in tables
- Complex64/128 data type support
- All integer types (int8 through int64, uint8 through uint32)
- Scaled integer handling (BSCALE/BZERO)
- HIERARCH and CONTINUE keyword support
- Specification v0.2 with MEF, tables, extended data types
- 37 new tests for Phase 1 (243 total)

---

## [0.1.0] -- 2026-04-12

### Added
- Initial NOVA format specification (v0.1 draft)
- Zarr v3 container with chunk-based access (container.py)
- Structured WCS using JSON-LD (wcs.py) -- 15+ projections, FITS round-trip
- Bidirectional FITS converter (fits_converter.py)
- W3C PROV-DM provenance (provenance.py)
- SHA-256 chunk integrity (integrity.py)
- JSON Schema validation (validation.py) -- 5 schema files
- ML-native tensor export (ml.py) -- 5 normalizations, PyTorch/JAX
- Integrated math operations (math.py) -- statistics, convolution, detection,
  aperture photometry, image stacking
- Display and plotting tools (visualization.py) -- 8 display functions
- Performance benchmarking (benchmarks.py) -- 5-format comparison
- High-performance binary I/O (fast_io.py)
- CLI with convert, info, validate, benchmark subcommands (cli.py)
- Benchmark plot generation (plots.py)
- 206 initial tests
