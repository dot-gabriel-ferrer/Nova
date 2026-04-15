# NOVA Format Specification -- Version 1.0 (Release Candidate)

**Status:** Release Candidate
**Date:** 2026-04-12
**Authors:** NOVA Project Contributors
**License:** CC-BY-4.0

---

## 1. Introduction

### 1.1 Purpose

This document specifies the NOVA (Next-generation Open Volumetric Archive) format,
a cloud-native scientific data format designed for professional astronomy. NOVA is
intended to succeed FITS (Flexible Image Transport System) as the international
standard for astronomical data interchange.

### 1.2 Scope

This specification defines:
- The binary container format and chunk index structure
- The metadata encoding using JSON-LD
- The World Coordinate System (WCS) schema
- The provenance model based on W3C PROV-DM
- The integrity verification mechanism
- The FITS compatibility layer
- Image processing, photometry, and spectral analysis pipelines
- Coordinate transforms and catalog operations
- A declarative pipeline framework with operation tracking
- Astrometry, photometry, and spectroscopy reduction pipelines

### 1.3 Terminology

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be
interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

### 1.4 Normative References

- [Zarr v3 Specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html)
- [JSON-LD 1.1](https://www.w3.org/TR/json-ld11/)
- [W3C PROV-DM](https://www.w3.org/TR/prov-dm/)
- [FITS Standard 4.0](https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf)
- [IVOA UCD1+](https://www.ivoa.net/documents/UCD1+/)
- [WCS Paper I-IV (Greisen & Calabretta)](https://fits.gsfc.nasa.gov/fits_wcs.html)
- [Horne 1986, PASP 98, 609](https://ui.adsabs.harvard.edu/abs/1986PASP...98..609H) -- Optimal extraction
- [Alard & Lupton 1998](https://ui.adsabs.harvard.edu/abs/1998ApJ...503..325A) -- Image subtraction
- [VOTable 1.4](https://www.ivoa.net/documents/VOTable/) -- IVOA table format
- [SAMP 1.3](https://www.ivoa.net/documents/SAMP/) -- Simple Application Messaging Protocol

---

## 2. Design Invariants

All implementations MUST satisfy these invariants. No specification revision
SHALL violate them without unanimous Working Group approval.

| ID | Invariant | Requirement |
|---|---|---|
| INV-1 | BACKWARD_COMPAT | Lossless FITS<->NOVA conversion for all valid FITS files |
| INV-2 | CLOUD_FIRST | Chunk index MUST reside in first 8KB; any region accessible in <=2 HTTP requests |
| INV-3 | HUMAN_READABLE | JSON-LD metadata manifest MUST be readable with any text editor |
| INV-4 | INTEGRITY_BY_DEFAULT | Every chunk MUST have a SHA-256 hash in the index |
| INV-5 | PROV_MANDATORY | W3C PROV-DM provenance REQUIRED for reduced/calibrated data |
| INV-6 | PARALLEL_WRITE | Format MUST support concurrent writes without global locks |
| INV-7 | ML_NATIVE | Native float16/BFloat16 support with standardized normalization metadata |

---

## 3. Layer 1 -- Container & Access

### 3.1 Base Container

A NOVA file MUST be a valid Zarr v3 store. The store MAY be:
- A directory on a POSIX filesystem (`.nova.zarr/`)
- A ZIP archive (`.nova.zip`)
- An object store accessible via HTTP (S3, GCS, Azure Blob)

### 3.2 Chunk Index

The chunk index MUST be stored at the root of the Zarr store in a file
named `nova_index.json`. This file MUST:

1. Be less than 8,192 bytes (8KB) for the core index
2. Contain byte offsets and sizes for all data chunks
3. Include SHA-256 hashes for each chunk (see section 6)

```json
{
  "@context": "https://nova-astro.org/v0.1/context.jsonld",
  "@type": "nova:ChunkIndex",
  "nova:version": "0.1.0",
  "nova:created": "2026-01-15T12:00:00Z",
  "nova:chunks": [
    {
      "nova:path": "data/science/c.0.0",
      "nova:offset": 8192,
      "nova:size": 1048576,
      "nova:sha256": "e3b0c44298fc1c149afbf4c8996fb924..."
    }
  ]
}
```

### 3.3 HTTP Range Access

Implementations SHOULD support HTTP Range requests to enable cloud-native access.
A client MUST be able to access any data region with at most 2 HTTP requests:

1. **Request 1:** Fetch `nova_index.json` (<=8KB)
2. **Request 2:** Fetch the specific chunk(s) using byte range from the index

### 3.4 File Extension

NOVA files MUST use one of the following extensions:
- `.nova.zarr` -- Directory store
- `.nova.zip` -- ZIP-archived store

### 3.5 Magic Bytes

When stored as a single-file archive, the first 8 bytes MUST be:
```
\x89NOVA\r\n\x1a\n
```
(Inspired by the PNG signature for file type detection and corruption detection.)

---

## 4. Layer 2 -- Compression & Encoding

### 4.1 Byte Order

All numeric data MUST be stored in little-endian byte order.

**Rationale (adoption decision):** This breaks with FITS big-endian convention
but aligns with all modern hardware (x86, ARM, RISC-V). The performance penalty
of big-endian on modern CPUs is measurable and unnecessary.

### 4.2 Compression Codecs

NOVA defines three compression tiers:

| Tier | Codec | Use Case | Zarr Codec ID |
|---|---|---|---|
| Scientific | ZSTD level 3 | Default for science data | `zstd` |
| Speed | LZ4 | Real-time pipelines | `lz4` |
| Preview | JPEG-XL | Thumbnail/preview images | `jxl` |

- Scientific data MUST use a lossless codec (ZSTD or LZ4).
- Preview images MAY use lossy compression (JPEG-XL) if the original
  lossless data is also present in the store.
- Uncompressed storage is permitted for small arrays (< 4KB).

### 4.3 Supported Data Types

| Type | Bytes | Zarr dtype | Notes |
|---|---|---|---|
| int8 | 1 | `int8` | |
| int16 | 2 | `int16` | |
| int32 | 4 | `int32` | |
| int64 | 8 | `int64` | |
| uint8 | 1 | `uint8` | |
| uint16 | 2 | `uint16` | |
| uint32 | 4 | `uint32` | |
| float16 | 2 | `float16` | ML-native (INV-7) |
| bfloat16 | 2 | `bfloat16` | ML-native (INV-7) |
| float32 | 4 | `float32` | |
| float64 | 8 | `float64` | Default for science |
| complex64 | 8 | `complex64` | Visibility data |
| complex128 | 16 | `complex128` | |

### 4.4 Chunking Strategy

Chunk sizes SHOULD be chosen to balance:
- Cloud access granularity (target: 1-4 MB per chunk)
- Compression ratio (larger chunks compress better)
- Memory footprint (smaller chunks reduce peak memory)

Default chunk shape for 2D images: `(512, 512)` for float64 data (2 MB per chunk).

---

## 5. Layer 3 -- Scientific Data

### 5.1 Array Groups

A NOVA store organizes data into Zarr groups:

```
observation.nova.zarr/
+-- nova_index.json          # Chunk index (section 3.2)
+-- nova_metadata.json       # Root metadata (section 7)
+-- data/
|   +-- science/             # Primary science array
|   |   +-- zarr.json
|   |   +-- c.<chunk_keys>
|   +-- uncertainty/         # Co-located uncertainty plane
|   |   +-- zarr.json
|   |   +-- c.<chunk_keys>
|   +-- mask/                # Data quality mask
|   |   +-- zarr.json
|   |   +-- c.<chunk_keys>
|   +-- preview/             # JPEG-XL preview (optional)
|       +-- zarr.json
|       +-- c.<chunk_keys>
+-- wcs.json                 # WCS metadata (section 8)
```

### 5.2 Science Array

The primary science data MUST be stored in `data/science/` as a Zarr v3 array.

- The array dtype MUST be one of the types in section 4.3.
- The array MUST have at least 2 dimensions for image data.
- Fill value MUST be `NaN` for floating-point types, `0` for integer types.

### 5.3 Uncertainty Plane

When present, the uncertainty array MUST:
- Have the same shape as the science array
- Be stored in `data/uncertainty/`
- Include metadata specifying the uncertainty type (standard deviation, variance, etc.)

### 5.4 Data Quality Mask

When present, the mask array MUST:
- Have the same shape as the science array
- Use uint8 or uint16 dtype with bitfield semantics
- Be stored in `data/mask/`

### 5.5 Multi-Extension Datasets

NOVA supports multi-extension datasets (MEF equivalent) through Zarr groups:

```
observation.nova.zarr/
+-- nova_metadata.json
+-- extensions.json            # Extension index
+-- data/
|   +-- science/               # Primary science data
+-- extensions/
|   +-- SCI/
|   |   +-- data/              # Image data
|   |   +-- wcs.json           # Extension-specific WCS
|   +-- ERR/
|   |   +-- data/
|   +-- DQ/
|   |   +-- data/
|   +-- VARIANCE/
|       +-- data/
+-- tables/
|   +-- CATALOG/               # Table data (one array per column)
|       +-- RA/
|       +-- DEC/
|       +-- MAG/
+-- wcs.json
```

#### 5.5.1 Extension Index

Multi-extension datasets MUST include an `extensions.json` file:

```json
{
  "@context": "https://nova-astro.org/v0.1/context.jsonld",
  "@type": "nova:MultiExtensionDataset",
  "nova:extensions": [
    {
      "nova:name": "SCI",
      "nova:extver": 1,
      "nova:has_data": true,
      "nova:shape": [2048, 4096],
      "nova:dtype": "float32",
      "nova:header": { "EXPTIME": 300.0, "BUNIT": "e-/s" }
    }
  ]
}
```

#### 5.5.2 Extension-Specific WCS

Each image extension MAY have its own WCS stored in
`extensions/<name>/wcs.json`.

#### 5.5.3 FITS MEF Mapping

| FITS Concept | NOVA Equivalent |
|---|---|
| Primary HDU | `data/science/` + root `wcs.json` |
| ImageHDU (`EXTNAME`) | `extensions/<EXTNAME>/data/` |
| BinTableHDU | `tables/<EXTNAME>/` |
| `EXTVER` | `nova:extver` in `extensions.json` |

### 5.6 Table Data

NOVA stores tabular data as collections of column arrays within a Zarr
group, enabling efficient columnar access (read single columns without
loading the full table).

#### 5.6.1 Table Storage

Each table is a Zarr group under `tables/<name>/`, with one Zarr array
per column.

#### 5.6.2 Table Metadata

Tables MUST be described in `tables.json` at the store root:

```json
{
  "@context": "https://nova-astro.org/v0.1/context.jsonld",
  "@type": "nova:TableCollection",
  "nova:tables": [
    {
      "@type": "nova:Table",
      "nova:name": "CATALOG",
      "nova:nrows": 1000,
      "nova:columns": [
        {
          "nova:name": "RA",
          "nova:dtype": "float64",
          "nova:length": 1000,
          "nova:unit": "deg",
          "nova:ucd": "pos.eq.ra"
        }
      ]
    }
  ]
}
```

#### 5.6.3 Column Metadata

Each column SHOULD include:
- `nova:unit` -- Physical unit (IVOA VOUnits syntax)
- `nova:ucd` -- IVOA Unified Content Descriptor
- `nova:description` -- Human-readable column description

### 5.7 Scaled Integers (BSCALE/BZERO)

When converting from FITS, implementations MUST handle BSCALE/BZERO:

- `BSCALE=1, BZERO=32768` -> convert int16 to uint16
- `BSCALE=1, BZERO=2147483648` -> convert int32 to uint32
- Other values -> apply `physical = BSCALE * pixel + BZERO`

---

## 6. Layer 4 -- Integrity

### 6.1 Chunk Hashes

Every data chunk MUST have a SHA-256 hash stored in the chunk index.
Implementations MUST verify chunk integrity on read by default.

Verification MAY be disabled for performance-critical batch processing
by passing an explicit `verify=False` parameter.

### 6.2 Metadata Hash

The `nova_metadata.json` file MUST include a SHA-256 hash of itself
(computed with the hash field set to a zero-length string).

### 6.3 Future: Cryptographic Signatures

A future revision of this specification will define:
- Ed25519 signatures for datasets
- Certificate chain for institutional provenance
- Timestamping for archival guarantees

---

## 7. Layer 4 -- Metadata

### 7.1 Root Metadata

Every NOVA store MUST contain a `nova_metadata.json` file at the root
with the following structure:

```json
{
  "@context": "https://nova-astro.org/v0.1/context.jsonld",
  "@type": "nova:Observation",
  "nova:version": "0.1.0",
  "nova:created": "2026-01-15T12:00:00Z",
  "nova:instrument": {
    "@type": "nova:Instrument",
    "nova:name": "DECam",
    "nova:telescope": "Blanco 4m",
    "nova:observatory": "CTIO"
  },
  "nova:data_level": "L1",
  "nova:content_hash": "sha256:abc123..."
}
```

### 7.2 Data Levels

| Level | Description | Provenance Required? |
|---|---|---|
| L0 | Raw data | NO |
| L1 | Calibrated/reduced | YES (INV-5) |
| L2 | Derived products | YES (INV-5) |
| L3 | Catalog/value-added | YES (INV-5) |

### 7.3 JSON-LD Context

All NOVA metadata MUST be valid JSON-LD. The `@context` field MUST
reference the NOVA vocabulary. NOVA uses JSON-LD to:

1. Ensure metadata is machine-readable with semantic meaning
2. Enable federation with IVOA vocabularies (UCDs, data models)
3. Allow custom extensions without namespace collisions

---

## 8. Layer 5 -- World Coordinate System (WCS)

### 8.1 Overview

The WCS in NOVA replaces FITS keyword-based WCS with a structured JSON-LD
object. This is the single most impactful change from FITS.

**Design rationale:** FITS WCS uses ~40 free-text keywords (CRPIX, CRVAL,
CTYPE, CD matrix, etc.) with complex parsing rules, multiple conventions
(CD vs CDELT+CROTA, PC matrix, SIP distortion), and no schema validation.
NOVA encodes the same information as a typed, validated JSON object.

### 8.2 WCS JSON-LD Schema

The WCS MUST be stored in `wcs.json` at the root of the NOVA store.
See `schemas/wcs.schema.json` for the normative JSON Schema.

### 8.3 WCS Structure

```json
{
  "@context": "https://nova-astro.org/v0.1/context.jsonld",
  "@type": "nova:WCS",
  "nova:naxes": 2,
  "nova:axes": [
    {
      "@type": "nova:CelestialAxis",
      "nova:index": 0,
      "nova:ctype": "RA---TAN",
      "nova:crpix": 2048.0,
      "nova:crval": 150.1191666667,
      "nova:unit": "deg",
      "nova:ucd": "pos.eq.ra"
    },
    {
      "@type": "nova:CelestialAxis",
      "nova:index": 1,
      "nova:ctype": "DEC--TAN",
      "nova:crpix": 2048.0,
      "nova:crval": 2.2058333333,
      "nova:unit": "deg",
      "nova:ucd": "pos.eq.dec"
    }
  ],
  "nova:transform": {
    "@type": "nova:AffineTransform",
    "nova:cd_matrix": [
      [-7.305555555556e-05, 0.0],
      [0.0, 7.305555555556e-05]
    ],
    "nova:pixel_scale": {
      "nova:value": 0.263,
      "nova:unit": "arcsec/pixel"
    }
  },
  "nova:celestial_frame": {
    "@type": "nova:CelestialFrame",
    "nova:system": "ICRS",
    "nova:equinox": 2000.0
  },
  "nova:projection": {
    "@type": "nova:GnomonicProjection",
    "nova:code": "TAN",
    "nova:name": "Gnomonic (Tangent Plane)"
  }
}
```

### 8.4 Axis Types

| Type | Description | Required Fields |
|---|---|---|
| `nova:CelestialAxis` | RA/Dec or Galactic lon/lat | ctype, crpix, crval, unit |
| `nova:SpectralAxis` | Wavelength, frequency, energy | ctype, crpix, crval, unit, restfrq (optional) |
| `nova:TemporalAxis` | Time axis | ctype, crpix, crval, unit, timesys |
| `nova:StokesAxis` | Polarization | ctype, stokes_params |
| `nova:LinearAxis` | Generic linear axis | ctype, crpix, crval, unit |

### 8.5 Coordinate Transforms

NOVA supports these transform types:

| Type | Description |
|---|---|
| `nova:AffineTransform` | CD matrix (general linear) |
| `nova:SIPDistortion` | Simple Imaging Polynomial |
| `nova:TPVDistortion` | Tangent Plane polynomial (TPV) |
| `nova:CompositeTransform` | Chain of transforms |

### 8.6 FITS WCS Compatibility

The FITS converter (section 10) MUST map all standard FITS WCS keywords to the
NOVA WCS schema. The mapping is bidirectional:

| FITS Keyword | NOVA Path |
|---|---|
| `WCSAXES` | `nova:naxes` |
| `CRPIXn` | `nova:axes[n-1].nova:crpix` |
| `CRVALn` | `nova:axes[n-1].nova:crval` |
| `CTYPEn` | `nova:axes[n-1].nova:ctype` |
| `CDi_j` | `nova:transform.nova:cd_matrix[i-1][j-1]` |
| `RADESYS` | `nova:celestial_frame.nova:system` |
| `EQUINOX` | `nova:celestial_frame.nova:equinox` |
| `CDELTn` | Computed from CD matrix diagonal |
| `CROTAn` | Computed from CD matrix rotation |
| `PCi_j` + `CDELTn` | Converted to CD matrix |

---

## 9. Layer 5 -- Provenance

### 9.1 W3C PROV-DM

NOVA uses W3C PROV-DM serialized as JSON-LD for provenance tracking.
See `schemas/provenance.schema.json` for the normative schema.

### 9.2 Provenance Requirements

- L0 (raw) data MAY omit provenance.
- L1+ (reduced/derived) data MUST include provenance (INV-5).
- Provenance MUST be stored in `provenance.json` at the store root.

### 9.3 Provenance Structure

```json
{
  "@context": [
    "https://nova-astro.org/v0.1/context.jsonld",
    "https://www.w3.org/ns/prov"
  ],
  "@type": "prov:Bundle",
  "prov:entity": [
    {
      "@id": "nova:entity/raw-frame-001",
      "@type": "prov:Entity",
      "nova:data_level": "L0"
    },
    {
      "@id": "nova:entity/calibrated-frame-001",
      "@type": "prov:Entity",
      "nova:data_level": "L1",
      "prov:wasDerivedFrom": { "@id": "nova:entity/raw-frame-001" }
    }
  ],
  "prov:activity": [
    {
      "@id": "nova:activity/bias-subtraction",
      "@type": "prov:Activity",
      "prov:used": { "@id": "nova:entity/raw-frame-001" },
      "prov:generated": { "@id": "nova:entity/calibrated-frame-001" },
      "nova:software": "nova-py v0.1.0",
      "nova:parameters": {
        "method": "median_combine",
        "sigma_clip": 3.0
      }
    }
  ]
}
```

---

## 10. FITS Compatibility Layer

### 10.1 FITS to NOVA Conversion

A conforming NOVA implementation MUST provide a FITS->NOVA converter that:

1. Preserves all header keywords as JSON-LD metadata
2. Converts WCS keywords to the structured WCS schema (section 8)
3. Converts BITPIX-encoded data to the appropriate Zarr dtype
4. Preserves FITS extensions as separate Zarr groups
5. Generates SHA-256 hashes for all data chunks
6. Stores the original FITS header verbatim in `fits_origin/header.txt`

### 10.2 NOVA to FITS Conversion

A conforming NOVA implementation MUST provide a NOVA->FITS converter that:

1. Maps all NOVA metadata back to FITS header keywords
2. Converts the WCS JSON-LD to standard FITS WCS keywords
3. Converts Zarr arrays to FITS HDU data with correct BITPIX
4. Handles float16/BFloat16 by upscaling to float32 (with warning)
5. Preserves provenance as HISTORY/COMMENT cards

### 10.3 FITS BITPIX Mapping

| FITS BITPIX | NOVA dtype |
|---|---|
| 8 | uint8 |
| 16 | int16 |
| 32 | int32 |
| 64 | int64 |
| -32 | float32 |
| -64 | float64 |

---

## 11. Remote Access (v0.3)

### 11.1 Remote Store Protocol

A conforming NOVA implementation SHOULD support opening stores from remote URLs.
The following protocols are RECOMMENDED:

| Protocol | URI Scheme | Backend |
|----------|-----------|---------|
| HTTP/HTTPS | `http://`, `https://` | fsspec HTTPFileSystem |
| Amazon S3 | `s3://` | s3fs via fsspec |
| Google Cloud Storage | `gs://` | gcsfs via fsspec |
| Azure Blob Storage | `az://`, `abfs://` | adlfs via fsspec |

### 11.2 Lazy Chunk Loading

Remote stores MUST support lazy chunk loading.  When a user accesses a
slice of a remote array (e.g. `ds.data[1000:1100, 2000:2100]`), only the
chunks that overlap with the requested region SHALL be fetched from the
remote server.  This is possible because:

1. The Zarr v3 chunk layout maps array indices to chunk file paths
2. The chunk index (`nova_index.json`) provides chunk offsets and sizes
3. HTTP Range requests or equivalent protocol features fetch individual chunks

### 11.3 Authentication

Remote stores SHOULD support authenticated access via `storage_options`
parameters forwarded to the filesystem constructor.  Supported mechanisms
include:

- AWS IAM credentials (key, secret, session token)
- OAuth2 bearer tokens
- Google service account JSON keys
- Azure shared access signatures (SAS)

---

## 12. Batch Migration (v0.3)

### 12.1 Migration Tool Requirements

A conforming NOVA implementation SHOULD provide a batch migration tool that:

1. Recursively discovers FITS files in a source directory
2. Converts each FITS file to a NOVA store in the destination directory
3. Preserves the directory structure
4. Supports parallel conversion (multiple worker processes)
5. Supports dry-run mode (analysis without writing)
6. Supports incremental mode (skip already converted files)
7. Optionally verifies round-trip fidelity by comparing array data

### 12.2 Migration Report

After conversion, the tool MUST produce a report containing:

- Total files found, converted, skipped, and failed
- Per-file size before and after conversion
- Elapsed time
- Verification status (if requested)

---

## 13. Streaming I/O (v0.3)

### 13.1 Append-Mode Writes

A conforming NOVA implementation SHOULD support appending data frames to an
existing array along a specified axis.  This enables time-series use cases
where data arrives incrementally.

The implementation MUST:

1. Resize the existing Zarr array to accommodate new frames
2. Write new data into the extended region
3. Update metadata (`nova:total_frames`, timestamps) on close
4. Support buffered writes to reduce disk I/O overhead

### 13.2 StreamWriter Interface

The recommended interface for streaming writes is:

```python
writer = open_appendable(store_path, frame_shape=(H, W))
for frame in data_source:
    writer.append(frame)
writer.close()  # flushes buffer, writes metadata
```

---

## 14. Pipeline Adapters (v0.3)

### 14.1 Astropy CCDData Adapter

A conforming NOVA implementation SHOULD provide bidirectional conversion
between NOVA datasets and astropy CCDData objects:

- `to_ccddata(nova_path)` -> CCDData with data, uncertainty, mask, WCS
- `from_ccddata(ccd, nova_path)` -> NOVA store

### 14.2 Astropy NDData Adapter

A lightweight adapter for astropy NDData (no unit requirement):

- `to_nddata(nova_path)` -> NDData with data, uncertainty, mask

### 14.3 HDUList Adapter

Conversion to in-memory astropy HDUList for compatibility with tools that
expect FITS HDU objects:

- `nova_to_hdulist(nova_path)` -> HDUList

---

## 15. Image Processing (v0.4)

### 15.1 PSF Modelling

Implementations MUST provide point-spread-function fitting with at least
the Moffat and Gaussian analytical profiles.

#### 15.1.1 Moffat PSF

The Moffat profile is defined as:

    I(r) = amplitude * (1 + ((x-x0)^2 + (y-y0)^2) / gamma^2)^(-alpha)

A conforming implementation MUST accept the following parameters:

| Parameter | Type | Description |
|---|---|---|
| image | ndarray (2D) | Input image containing the source |
| x0 | float | Initial x centre estimate |
| y0 | float | Initial y centre estimate |
| box_size | int | Fitting box side length (default: 21) |

The fit result MUST include at minimum: `x0`, `y0`, `amplitude`, `gamma`,
`alpha`, and `fwhm`.

#### 15.1.2 Gaussian PSF

The 2-D circular Gaussian profile is defined as:

    I(r) = amplitude * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))

Parameter requirements are the same as section 15.1.1. The fit result
MUST include `x0`, `y0`, `amplitude`, `sigma`, and `fwhm`.

#### 15.1.3 Multi-Gaussian PSF

Implementations SHOULD support a composite PSF model consisting of
multiple concentric Gaussian components with independent amplitudes and
sigma values. This is useful for representing the core + halo structure
of realistic PSFs.

Required parameters:

| Parameter | Type | Description |
|---|---|---|
| x, y | ndarray | Evaluation coordinates |
| amplitudes | sequence of float | Per-component amplitudes |
| x0, y0 | float | Shared centre position |
| sigmas | sequence of float | Per-component sigma values |

### 15.2 Image Registration and Alignment

#### 15.2.1 Cross-Correlation Shift

Implementations MUST support sub-pixel image registration via
cross-correlation. The shift computation MUST accept an `upsample_factor`
parameter (default: 10) that controls sub-pixel precision. The result
MUST be a (dy, dx) shift tuple in pixels.

#### 15.2.2 Feature-Based Alignment

Implementations SHOULD support feature-based alignment using DAOFIND-style
source detection and triangle matching. Parameters:

| Parameter | Type | Description |
|---|---|---|
| reference | ndarray (2D) | Reference image |
| target | ndarray (2D) | Image to align |
| max_features | int | Maximum source count (default: 200) |
| match_threshold | float | Matching similarity threshold (default: 0.7) |

The result MUST be the reprojected target array.

#### 15.2.3 WCS-Based Alignment

Implementations SHOULD support resampling one image onto the pixel grid
of another using their respective WCS solutions. The resampling order
(interpolation degree) MUST be configurable with a default of 1 (bilinear).

### 15.3 Image Subtraction

Implementations MUST support difference imaging following the
Alard & Lupton (1998) kernel-matching algorithm.

| Parameter | Type | Description |
|---|---|---|
| science | ndarray (2D) | Science frame |
| reference | ndarray (2D) | Reference (template) frame |
| kernel_size | int | Convolution kernel side (default: 11) |
| n_basis | int | Number of Gaussian basis widths (default: 3) |
| deg_spatial | int | Degree of spatial polynomial variation (default: 2) |

The result MUST be the difference image (science - convolved reference).
Implementations SHOULD also provide a simple scaled subtraction for cases
where kernel matching is unnecessary.

### 15.4 CCD Calibration Pipeline

#### 15.4.1 Bias Subtraction

Implementations MUST subtract a master bias frame from a raw image.
The bias and data arrays MUST have the same shape.

#### 15.4.2 Dark Current Correction

Implementations MUST subtract a scaled dark frame. The scaling MUST use
the ratio of science exposure time to dark exposure time:

    corrected = data - dark * (exposure_data / exposure_dark)

#### 15.4.3 Flat Field Correction

Implementations MUST divide by a normalized flat field. Pixels in the
flat below a configurable floor (default: 0.01) MUST be clipped to
avoid division-by-zero artifacts.

#### 15.4.4 Overscan Correction

Implementations SHOULD support overscan subtraction with configurable
method. Supported methods:

| Method | Description |
|---|---|
| `mean` | Mean of the overscan region |
| `median` | Median of the overscan region (default) |
| `polynomial` | Polynomial fit along the overscan columns |

#### 15.4.5 Fringe Correction

Implementations SHOULD support construction of a fringe map from a stack
of images and subtraction of the scaled fringe pattern.

### 15.5 Bad Pixel Handling

#### 15.5.1 Bad Pixel Mask Construction

Implementations MUST support building a bad pixel mask from a set of
dark frames and flat frames. A pixel is flagged as bad if:

- Its dark current exceeds `dark_thresh` times the median absolute
  deviation (MAD) above the median of the dark stack, OR
- Its flat field response is below `flat_low` or above `flat_high`
  (defaults: 0.5, 1.5).

#### 15.5.2 Bad Pixel Interpolation

Implementations MUST support interpolation over flagged bad pixels.
Supported methods: `linear` (default), `nearest`.

---

## 16. Photometry (v0.4)

### 16.1 PSF Photometry

Implementations MUST support PSF-fitting photometry for point sources.

| Parameter | Type | Description |
|---|---|---|
| image | ndarray (2D) | Science image |
| positions | ndarray (N, 2) | Source positions (x, y) |
| psf_model | ndarray (2D) | Normalized PSF stamp |
| box_size | int | Fitting box side (default: 21) |
| n_iters | int | Fitting iterations (default: 3) |

Each source result MUST include: `x_fit`, `y_fit`, `flux`, and `flux_err`.

### 16.2 Extended Source Photometry

#### 16.2.1 Petrosian Radius

Implementations MUST compute the Petrosian radius defined as the radius
where the ratio of the local surface brightness to the mean interior
surface brightness falls to a specified threshold `eta` (default: 0.2).

#### 16.2.2 Kron Radius

Implementations MUST compute the first-moment (Kron) radius:

    r_kron = k * sum(r * I(r)) / sum(I(r))

where `k` is the Kron factor (default: 2.5) and the sums are evaluated
over pixels within an initial circular aperture.

#### 16.2.3 Isophotal Photometry

Implementations MUST support isophotal photometry, summing flux within a
connected region above a surface brightness threshold. The result MUST
include: `flux`, `area_pix`, `x_centroid`, `y_centroid`.

### 16.3 Photometric Calibration

#### 16.3.1 Zero-Point Determination

Implementations MUST compute a photometric zero-point from a set of
instrumental magnitudes and catalog magnitudes. Sigma-clipping SHOULD
be applied to reject outliers.

| Parameter | Type | Description |
|---|---|---|
| inst_mags | ndarray | Instrumental magnitudes |
| catalog_mags | ndarray | Reference catalog magnitudes |
| weights | ndarray or None | Optional per-source weights |

The result MUST include `zeropoint` and `zeropoint_err`.

#### 16.3.2 Zero-Point Application

Implementations MUST provide a function to apply a zero-point offset:

    calibrated_mag = inst_mag + zeropoint

Uncertainty propagation SHOULD be supported when instrumental errors
are provided.

#### 16.3.3 Atmospheric Extinction Correction

Implementations MUST correct for atmospheric extinction:

    m_corrected = m_observed - k * X

where `k` is the extinction coefficient and `X` is the airmass.

#### 16.3.4 Color-Term Correction

Implementations SHOULD support color-term correction:

    m_corrected = m_inst + color_term * color_index

### 16.4 Crowded-Field Photometry

#### 16.4.1 Source Deblending

Implementations MUST support simultaneous PSF fitting of blended sources
within a fitting box. The algorithm MUST iterate to convergence, fitting
all overlapping PSFs jointly.

#### 16.4.2 Iterative PSF Subtraction

Implementations MUST support sequential subtraction of fitted PSFs from
the image. The return value MUST include both the residual image and the
per-source photometry results.

---

## 17. Spectral Analysis (v0.4)

### 17.1 Wavelength Calibration

#### 17.1.1 Polynomial Wavelength Solution

Implementations MUST support fitting a polynomial wavelength solution to
identified arc lamp lines. The fit is:

    wavelength = sum(c_i * pixel^i,  i = 0..order)

| Parameter | Type | Description |
|---|---|---|
| pixel_positions | ndarray | Pixel coordinates of identified lines |
| known_wavelengths | ndarray | Laboratory wavelengths of those lines |
| order | int | Polynomial degree (default: 3) |

The result MUST include `coefficients`, `rms_residual`, and `n_lines`.

#### 17.1.2 Wavelength Application

Implementations MUST provide a function to evaluate the wavelength solution
at arbitrary pixel positions, returning calibrated wavelength values.

### 17.2 Sky Subtraction

#### 17.2.1 2-D Sky Subtraction

Implementations MUST support subtracting a sky model from a 2-D spectral
image. The sky model is constructed from designated sky regions along the
spatial axis. Supported methods: `median` (default), `polynomial`.

#### 17.2.2 1-D Sky Subtraction

Implementations MUST support subtracting a scaled 1-D sky spectrum from
an extracted 1-D science spectrum.

### 17.3 Radial Velocity

#### 17.3.1 Cross-Correlation

Implementations MUST support radial velocity measurement via
cross-correlation of an observed spectrum against a template. The velocity
search grid MUST be configurable:

| Parameter | Type | Description |
|---|---|---|
| observed | ndarray | Observed flux array |
| template | ndarray | Template flux array |
| wavelengths | ndarray | Wavelength array (shared grid) |
| v_range | tuple (float, float) | Velocity search bounds in km/s (default: -500, 500) |
| v_step | float | Velocity step in km/s (default: 1.0) |

The result MUST include `rv_kms` (best-fit velocity), `rv_err_kms`, and
`cc_peak` (peak cross-correlation value).

#### 17.3.2 Barycentric Correction

Implementations SHOULD support applying a barycentric velocity correction
to a wavelength array.

### 17.4 Emission Line Fitting

#### 17.4.1 Single-Line Fitting

Implementations MUST support fitting a single emission or absorption line
with either a Gaussian or Voigt profile.

| Parameter | Type | Description |
|---|---|---|
| wavelengths | ndarray | Wavelength array |
| flux | ndarray | Flux array |
| center_guess | float | Initial wavelength estimate for the line |
| profile | str | `"gaussian"` (default) or `"voigt"` |

The result MUST include `center`, `amplitude`, `sigma` (or equivalent
width parameters), and `fwhm`.

#### 17.4.2 Multi-Line Fitting

Implementations SHOULD support simultaneous fitting of multiple lines in
a single spectrum, returning a list of per-line fit results.

#### 17.4.3 Equivalent Width

Implementations MUST support computing the equivalent width of a spectral
feature relative to a provided continuum estimate. The line integration
range MUST be specified explicitly. When variance is provided, the
result SHOULD include `ew_err`.

### 17.5 Echelle Support

#### 17.5.1 Order Tracing

Implementations SHOULD support tracing individual echelle orders across
the 2-D detector image. The trace algorithm MUST walk column-by-column (or
at a configurable step interval), locating the flux centroid within a
search window.

| Parameter | Type | Description |
|---|---|---|
| image_2d | ndarray (2D) | Echelle spectral image |
| start_col | int | Starting column index |
| start_row | int | Starting row (initial centroid) |
| trace_step | int | Column step between centroid measurements (default: 10) |
| search_width | int | Half-width of the centroid search window (default: 5) |

#### 17.5.2 Order Extraction

Implementations SHOULD support extracting 1-D spectra from each echelle
order given a set of trace arrays and an aperture half-width.

#### 17.5.3 Order Merging

Implementations SHOULD support merging overlapping orders into a single
continuous spectrum. In overlap regions the implementation MUST support
at least `weighted` and `average` combination methods.

---

## 18. Coordinate Transforms (v0.4)

### 18.1 Tangent-Plane Projection

Implementations MUST support the gnomonic (TAN) projection as the baseline
celestial projection. Forward projection (world -> intermediate) and
deprojection (intermediate -> world) MUST both be provided.

### 18.2 SIP Distortion

Implementations MUST support the Simple Imaging Polynomial (SIP) convention
for astrometric distortion as defined in Shupe et al. (2005).

#### 18.2.1 Forward SIP Transform

The forward SIP distortion maps undistorted intermediate coordinates
(u, v) to distorted coordinates (u', v') via polynomial correction terms
stored in A and B coefficient matrices:

    u' = u + f(u, v; A)
    v' = v + g(u, v; B)

where f and g are polynomials up to a configurable order.

| Parameter | Type | Description |
|---|---|---|
| u, v | float or ndarray | Undistorted intermediate pixel offsets |
| a_coeffs | dict[(int,int), float] | A polynomial coefficients |
| b_coeffs | dict[(int,int), float] | B polynomial coefficients |
| a_order, b_order | int | Polynomial orders |

#### 18.2.2 Inverse SIP Transform

The inverse SIP transform maps distorted coordinates back to undistorted
using AP and BP coefficient matrices. The parameter structure mirrors
section 18.2.1 with `ap_coeffs` and `bp_coeffs`.

#### 18.2.3 SIP Pixel-to-World and World-to-Pixel

Implementations MUST provide combined transforms that chain the SIP
distortion with the CD matrix and TAN projection to convert between
pixel coordinates and celestial (RA, Dec) coordinates.

### 18.3 TPV Distortion

Implementations MUST support the Tangent Plane Polynomial (TPV) distortion
convention used in many ground-based survey pipelines.

#### 18.3.1 TPV Forward Transform

The TPV distortion maps intermediate coordinates (xi, eta) to corrected
coordinates via PV1 and PV2 coefficient dictionaries keyed by term index.

#### 18.3.2 TPV Pixel-to-World and World-to-Pixel

Combined transforms MUST be provided. The world-to-pixel direction MUST
use iterative inversion (default: 20 iterations) since the TPV polynomial
is not analytically invertible.

### 18.4 Lookup Table Distortion

Implementations SHOULD support image distortion correction via sampled
lookup tables, as used in HST and JWST data products.

| Parameter | Type | Description |
|---|---|---|
| pixel_x, pixel_y | float or ndarray | Input pixel coordinates |
| distortion_table_x | ndarray (2D) | X correction lookup table |
| distortion_table_y | ndarray (2D) | Y correction lookup table |
| crpix_table | tuple (float, float) | Reference pixel for the table (default: 1, 1) |
| cdelt_table | tuple (float, float) | Pixel scale of the table (default: 1, 1) |

Interpolation between table grid points MUST use bilinear interpolation.

### 18.5 Frame Transforms

Implementations MUST support coordinate frame conversions between the
standard astronomical reference frames.

| Transform | Description |
|---|---|
| Equatorial -> Galactic | J2000 equatorial (RA, Dec) to Galactic (l, b) |
| Galactic -> Equatorial | Galactic (l, b) to J2000 equatorial (RA, Dec) |
| Equatorial -> Ecliptic | Equatorial to ecliptic (lon, lat); obliquity configurable (default: 23.439281 deg) |
| Ecliptic -> Equatorial | Ecliptic to equatorial; same obliquity convention |
| FK5 -> ICRS | FK5 to ICRS (identity at J2000 to first order; frame rotation applied) |

Implementations SHOULD also support epoch precession for equatorial
coordinates between arbitrary Julian epochs.

---

## 19. Catalog Operations (v0.4)

### 19.1 Cross-Matching

#### 19.1.1 Positional Cross-Match

Implementations MUST support positional cross-matching of two catalogs
using a configurable match radius (default: 1.0 arcsec). The algorithm
MUST find the nearest neighbor in catalog 2 for each source in catalog 1
within the search radius.

The result MUST include:
- `idx1` -- indices of matched sources in catalog 1
- `idx2` -- indices of matched sources in catalog 2
- `separations` -- angular separations in arcseconds
- `n_matches` -- total match count

#### 19.1.2 Self-Match

Implementations SHOULD support self-matching within a single catalog to
identify duplicate detections.

#### 19.1.3 Nearest-Neighbor Search

Implementations SHOULD provide a nearest-neighbor lookup that returns the
single closest match for each source, along with an optional maximum
radius cutoff.

### 19.2 Spatial Queries

#### 19.2.1 Cone Search

Implementations MUST support selecting sources within a circular region:

| Parameter | Type | Description |
|---|---|---|
| ra_catalog, dec_catalog | ndarray | Catalog positions in degrees |
| ra_center, dec_center | float | Cone centre in degrees |
| radius | float | Search radius in arcseconds |

The result MUST be a tuple of (indices, separations).

#### 19.2.2 Box Search

Implementations MUST support selecting sources within a rectangular region
defined by (ra_min, ra_max, dec_min, dec_max) in degrees. The result
MUST be an index array of selected sources.

#### 19.2.3 Polygon Search

Implementations SHOULD support selecting sources within an arbitrary
spherical polygon defined by vertex arrays (ra_vertices, dec_vertices).

### 19.3 VOTable Support

#### 19.3.1 VOTable Reading

Implementations MUST support reading VOTable 1.4 files into an in-memory
dictionary of column arrays plus metadata.

#### 19.3.2 VOTable Writing

Implementations MUST support writing column arrays and metadata to a
VOTable 1.4 file using the TABLEDATA serialization.

#### 19.3.3 NOVA-to-VOTable Conversion

Implementations SHOULD provide direct conversion from a NOVA table group
(section 5.6) to a VOTable file, and the reverse.

### 19.4 SAMP Integration

Implementations SHOULD provide a SAMP (Simple Application Messaging
Protocol) client that enables interoperability with VO-aware applications
such as TOPCAT, Aladin, and DS9.

The SAMP client MUST support:
- Connecting to and disconnecting from a SAMP hub
- Broadcasting table data
- Sending spectrum and image references

### 19.5 Catalog Statistics

Implementations SHOULD provide utility functions for catalog analysis:

| Function | Description |
|---|---|
| healpix_index | Compute HEALPix indices for catalog positions |
| source_density | Compute source density (sources per square degree) |
| number_counts | Compute differential number counts per magnitude bin |
| magnitude_histogram | Compute magnitude distribution histogram |

---

## 20. Pipeline Framework (v0.5)

### 20.1 Overview

A conforming implementation MUST provide a declarative pipeline framework
that composes named processing steps into reproducible workflows. The
pipeline records a complete execution log with timestamps, checksums,
and parameter summaries for every step.

### 20.2 Pipeline Construction

A Pipeline MUST be constructed with:

| Parameter | Type | Required | Description |
|---|---|---|---|
| name | str | YES | Pipeline identifier (e.g. `"ccd_reduction"`) |
| version | str | NO | Version tag (default: `"1.0"`) |
| metadata | dict or None | NO | Arbitrary metadata (instrument, observer, etc.) |

### 20.3 Step Management

#### 20.3.1 Adding Steps

Steps MUST be added via `add_step`, which appends a named callable
to the pipeline. Each step receives a NumPy array as its first argument
and MUST return a NumPy array. Additional keyword arguments are captured
at definition time and forwarded at execution.

| Parameter | Type | Description |
|---|---|---|
| name | str | Unique step name |
| func | callable | Processing function: ndarray -> ndarray |
| description | str | Optional human-readable description |
| **params | any | Keyword arguments forwarded to func |

#### 20.3.2 Inserting Steps

Implementations MUST support inserting a step at an arbitrary position
(0-based index) via `insert_step`.

#### 20.3.3 Removing Steps

Implementations MUST support removing a step by name via `remove_step`.
If no step with the given name exists, a KeyError MUST be raised.

### 20.4 Pipeline Execution

When `run(data)` is called the pipeline MUST:

1. Record the ISO-8601 start timestamp.
2. For each step in order:
   a. Compute the SHA-256 hex digest of the input array bytes.
   b. Execute the step function with the array and its bound parameters.
   c. Compute the SHA-256 hex digest of the output array bytes.
   d. Record the step name, function name, parameter summary, input/output
      checksums, shapes, dtypes, start time, end time, and duration.
3. Record the overall end timestamp and total duration.
4. Store the complete log in the `log` attribute as a PipelineLog.

### 20.5 Step Logging

Each step execution MUST produce a StepLog record containing:

| Field | Type | Description |
|---|---|---|
| step_name | str | Name of the step |
| func_name | str | Qualified name of the callable |
| params_summary | dict | Serializable snapshot of parameters |
| input_sha256 | str | SHA-256 of input array bytes |
| output_sha256 | str | SHA-256 of output array bytes |
| input_shape | tuple of int | Shape of the input array |
| output_shape | tuple of int | Shape of the output array |
| input_dtype | str | Data type of the input |
| output_dtype | str | Data type of the output |
| started_at | str | ISO-8601 timestamp |
| ended_at | str | ISO-8601 timestamp |
| duration_seconds | float | Wall-clock duration |
| description | str | Optional step description |

### 20.6 Pipeline Log

The PipelineLog MUST aggregate all StepLog records and include:

| Field | Type | Description |
|---|---|---|
| pipeline_name | str | Pipeline name |
| version | str | Pipeline version |
| executed_at | str | ISO-8601 execution start |
| total_duration_seconds | float | Total wall-clock time |
| steps | list of StepLog | Ordered step records |
| metadata | dict | User-supplied metadata |

### 20.7 Serialization

PipelineLog and StepLog MUST support serialization to and from JSON
via `to_json()` and `from_json()` class methods. The JSON representation
MUST be deterministic (sorted keys, consistent formatting) to enable
diff-based comparison of pipeline runs.

---

## 21. Native Operations (v0.5)

### 21.1 Overview

A conforming implementation MUST provide tracked array operations that
automatically record every transformation applied to scientific data.
This enables full provenance reconstruction without relying on external
logging frameworks.

### 21.2 OperationHistory

The OperationHistory is an append-only log of OperationRecord entries.

- New records MUST be appended; existing records MUST NOT be modified
  or deleted.
- Implementations MUST support `len()` to query the number of records.
- The `records` property MUST return a read-only copy of the record list.

### 21.3 OperationRecord

Each tracked operation MUST produce an OperationRecord containing:

| Field | Type | Description |
|---|---|---|
| operation | str | Operation label (e.g. `"subtract"`, `"clip"`) |
| timestamp | str | ISO-8601 timestamp of the operation |
| input_sha256 | str | SHA-256 of the primary input array |
| output_sha256 | str | SHA-256 of the result array |
| input_shape | tuple of int | Shape of the input |
| output_shape | tuple of int | Shape of the output |
| params | dict | Operation-specific parameters |

### 21.4 Tracked Arithmetic

The following element-wise operations MUST be provided. Each accepts an
optional `history` parameter; when supplied, a record is automatically
appended.

| Function | Operation | Notes |
|---|---|---|
| op_add | data + operand | Scalar or array operand |
| op_subtract | data - operand | Scalar or array operand |
| op_multiply | data * operand | Scalar or array operand |
| op_divide | data / operand | Scalar or array operand |

All arithmetic functions MUST accept `data` (ndarray), `operand`
(ndarray or float), an optional `history`, and an optional `label`.

### 21.5 Tracked Clipping

`op_clip` MUST clip array values to a specified range [lower, upper].
Either bound MAY be None (unbounded). Parameters:

| Parameter | Type | Description |
|---|---|---|
| data | ndarray | Input array |
| lower | float or None | Minimum value |
| upper | float or None | Maximum value |

### 21.6 Tracked Masking

`op_mask_replace` MUST replace masked pixels with a fill value.

| Parameter | Type | Description |
|---|---|---|
| data | ndarray | Input array |
| mask | ndarray (bool) | Mask array (True = replace) |
| fill_value | float | Replacement value (default: 0.0) |

### 21.7 Tracked Normalization

`op_normalize` MUST normalize the array. Supported methods:

| Method | Formula |
|---|---|
| `minmax` | (data - min) / (max - min) |
| `zscore` | (data - mean) / std |

The `method` parameter MUST default to `"minmax"`.

### 21.8 Tracked Rebinning

`op_rebin` MUST rebin a 2-D array by an integer factor, averaging blocks
of `factor x factor` pixels. The input dimensions MUST be evenly
divisible by the factor.

### 21.9 Tracked Combine

`op_combine` MUST combine a list of arrays into a single array.
Supported methods:

| Method | Description |
|---|---|
| `mean` | Element-wise mean |
| `median` | Element-wise median (default) |
| `sum` | Element-wise sum |

### 21.10 JSON Serialization

OperationHistory MUST support `to_json()` and `from_json()` for lossless
round-trip serialization. The JSON format MUST be a list of record
dictionaries, each containing all fields from section 21.3.

---

## 22. Astrometry Pipeline (v0.5)

### 22.1 Centroid Extraction

Implementations MUST provide source detection and centroid extraction from
a 2-D image.

| Parameter | Type | Description |
|---|---|---|
| data | ndarray (2D) | Input image |
| fwhm | float | Expected PSF FWHM in pixels (default: 3.5) |
| threshold | float | Detection threshold in sigma above background (default: 5.0) |
| max_sources | int | Maximum sources to return (default: 500) |
| border | int | Exclusion border width in pixels (default: 10) |

The result MUST be an (N, 2) array of (x, y) centroid positions, sorted
by decreasing brightness.

### 22.2 Plate Solving

Implementations MUST support plate solving via geometric triangle matching
between detected source positions and a reference catalog.

| Parameter | Type | Description |
|---|---|---|
| detected | ndarray (N, 2) | Detected pixel positions |
| catalog_radec | ndarray (M, 2) | Reference RA/Dec in degrees |
| image_shape | tuple (int, int) | Image dimensions |
| pixel_scale_guess | float or None | Approximate pixel scale in deg/px |
| match_tol | float | Triangle match tolerance (default: 0.01) |
| n_bright | int | Number of brightest sources used for matching (default: 30) |

The result MUST include: `crpix`, `crval`, `cd_matrix`, `n_matches`,
and `rms_arcsec` (root-mean-square residual in arcseconds).

### 22.3 Proper Motion Correction

Implementations MUST support correcting catalog positions for proper
motion between two Julian epochs:

    ra_new  = ra  + pmra  * cos(dec) * (epoch_to - epoch_from)
    dec_new = dec + pmdec * (epoch_to - epoch_from)

where `pmra` and `pmdec` are in degrees per year.

### 22.4 Parallax Correction

Implementations MUST support correcting catalog positions for parallax
given the observer's barycentric position vector (in AU). The correction
uses the standard parallax formula relating angular displacement to the
observer-source geometry.

### 22.5 SIP Distortion Fitting

Implementations MUST support fitting SIP distortion coefficients given a
set of matched (pixel, sky) coordinate pairs and an initial linear WCS
solution (CRPIX, CRVAL, CD matrix).

| Parameter | Type | Description |
|---|---|---|
| detected_xy | ndarray (N, 2) | Detected pixel coordinates |
| catalog_radec | ndarray (N, 2) | Matched catalog RA/Dec |
| crpix | tuple (float, float) | Reference pixel |
| crval | tuple (float, float) | Reference sky position |
| cd_matrix | ndarray (2, 2) | Linear CD matrix |
| order | int | SIP polynomial order (default: 3) |

The result MUST include the A and B coefficient dictionaries.

### 22.6 Astrometric Residuals

Implementations MUST provide a function to compute astrometric residuals
between detected and catalog positions. The result MUST include per-source
separations and the aggregate RMS in arcseconds.

---

## 23. Photometry Pipeline (v0.5)

### 23.1 Multi-Aperture Photometry

Implementations MUST support photometry at multiple aperture radii
simultaneously, with configurable sky annulus.

| Parameter | Type | Description |
|---|---|---|
| data | ndarray (2D) | Science image |
| sources | ndarray (N, 2) | Source positions (x, y) |
| radii | list of float | Aperture radii in pixels |
| sky_inner | float or None | Inner sky annulus radius |
| sky_outer | float or None | Outer sky annulus radius |
| gain | float | Detector gain in e-/ADU (default: 1.0) |
| readnoise | float | Read noise in e- (default: 0.0) |

The result MUST include per-source, per-aperture flux and flux error,
plus the estimated sky level per source.

### 23.2 Zero-Point Calibration

Implementations MUST support computing a photometric zero-point by
comparing instrumental and catalog magnitudes.

| Parameter | Type | Description |
|---|---|---|
| inst_mag | ndarray | Instrumental magnitudes |
| catalog_mag | ndarray | Reference catalog magnitudes |
| sigma_clip | float | Sigma clipping threshold (default: 3.0) |
| max_iter | int | Maximum clipping iterations (default: 5) |

The result MUST include `zeropoint`, `zeropoint_err`, `n_used`, and
`n_rejected`.

### 23.3 Extinction Correction

Implementations MUST correct magnitudes for atmospheric extinction:

    m_corrected = m_observed - k_lambda * airmass

### 23.4 Differential Photometry

Implementations MUST support differential photometry relative to one or
more comparison stars.

| Parameter | Type | Description |
|---|---|---|
| target_flux | ndarray | Time-series flux of the target |
| comparison_flux | ndarray | Time-series flux of comparison star(s) |
| comparison_mag | float or None | Known magnitude of the comparison |

The result MUST include `diff_mag` and `diff_mag_err` arrays.

### 23.5 Growth Curve Analysis

Implementations MUST support computing a curve of growth -- the enclosed
flux as a function of aperture radius.

| Parameter | Type | Description |
|---|---|---|
| data | ndarray (2D) | Image data |
| x, y | float | Source centre coordinates |
| radii | list of float | Radii at which to measure flux |

The result MUST include `radii`, `flux`, and `normalized_flux` arrays.

### 23.6 Aperture Correction

Implementations SHOULD support computing an aperture correction from
photometry measured at a small and large aperture radius. Sigma-clipping
SHOULD be applied to the magnitude differences. The result MUST include
`apcor` and `apcor_err`.

### 23.7 Magnitude Conversion Utilities

Implementations SHOULD provide the following utility functions:

| Function | Description |
|---|---|
| flux_to_mag | Convert flux to magnitude given a zero-point |
| mag_to_flux | Convert magnitude to flux given a zero-point |
| ab_to_vega | Convert AB magnitude to Vega given an offset |
| vega_to_ab | Convert Vega magnitude to AB given an offset |
| limiting_magnitude | Estimate limiting magnitude from sky RMS, zero-point, and aperture size |
| color_term_correct | Apply a color-term correction |

---

## 24. Spectroscopy Pipeline (v0.5)

### 24.1 Optimal Extraction

Implementations MUST support optimal spectral extraction following the
variance-weighted algorithm of Horne (1986). This maximizes
signal-to-noise by weighting each spatial pixel by the known spatial
profile divided by its variance.

| Parameter | Type | Description |
|---|---|---|
| data_2d | ndarray (2D) | Rectified 2-D spectral image |
| trace | ndarray (1D) | Trace position as a function of column |
| profile | ndarray (2D) or None | Normalized spatial profile (auto-estimated if None) |
| variance_2d | ndarray (2D) or None | Variance image (estimated from data + gain/readnoise if None) |
| aperture_half | int | Half-width of extraction aperture in pixels (default: 5) |
| gain | float | Detector gain in e-/ADU (default: 1.0) |
| readnoise | float | Read noise in e- (default: 0.0) |

The result MUST include `flux` (1-D optimal spectrum) and `variance`
(1-D variance spectrum).

### 24.2 Continuum Fitting

Implementations MUST support fitting a smooth continuum to a 1-D spectrum
for normalization and continuum subtraction.

| Parameter | Type | Description |
|---|---|---|
| wavelength | ndarray | Wavelength array |
| flux | ndarray | Flux array |
| method | str | `"polynomial"` (default) or `"median_filter"` |
| order | int | Polynomial degree (default: 5, for polynomial method) |
| sigma_clip | float | Iterative rejection threshold (default: 3.0) |
| max_iter | int | Maximum rejection iterations (default: 5) |
| window | int or None | Window size for median filter method |

The result MUST be a 1-D continuum array of the same length as flux.

### 24.3 Telluric Correction

#### 24.3.1 Telluric Division

Implementations MUST support dividing a science spectrum by an observed
telluric standard transmission spectrum. A configurable minimum
transmission floor (default: 0.1) MUST be applied to avoid amplifying
noise in saturated telluric bands.

#### 24.3.2 Telluric Modelling

Implementations SHOULD support generating a synthetic telluric transmission
spectrum from a list of absorption bands specified as (wavelength_min,
wavelength_max) tuples with configurable depth.

### 24.4 Spectral Stacking

Implementations MUST support combining multiple 1-D spectra onto a common
wavelength grid.

| Parameter | Type | Description |
|---|---|---|
| wavelengths | list of ndarray | Per-spectrum wavelength arrays |
| fluxes | list of ndarray | Per-spectrum flux arrays |
| common_grid | ndarray or None | Output wavelength grid (auto-generated if None) |
| method | str | `"median"` (default), `"mean"`, or `"weighted_mean"` |
| variances | list of ndarray or None | Optional per-spectrum variance arrays |

The result MUST include `wavelength`, `flux`, and (when variances are
provided) `variance` arrays on the common grid.

### 24.5 Redshift Measurement

Implementations MUST support measuring spectroscopic redshift via
cross-correlation with a template spectrum over a configurable redshift
grid.

| Parameter | Type | Description |
|---|---|---|
| wavelength | ndarray | Observed wavelength array |
| flux | ndarray | Observed flux array |
| template_wavelength | ndarray | Template wavelength array |
| template_flux | ndarray | Template flux array |
| z_min | float | Minimum trial redshift (default: 0.0) |
| z_max | float | Maximum trial redshift (default: 1.0) |
| z_step | float | Redshift step size (default: 0.0001) |

The result MUST include `redshift` (best-fit), `redshift_err`,
`cc_peak`, and `cc_values` (full cross-correlation array).

### 24.6 Signal-to-Noise Estimation

Implementations MUST support estimating the signal-to-noise ratio of a
1-D spectrum.

| Parameter | Type | Description |
|---|---|---|
| flux | ndarray | Flux array |
| variance | ndarray or None | Variance array (if available) |
| method | str | `"der_snr"` (default) or `"variance"` |

The `der_snr` method estimates noise from the second derivative of the
flux array (DER_SNR algorithm) and does not require a variance input.
The `variance` method computes SNR = flux / sqrt(variance).

The result MUST include `snr_median` (median SNR across the spectrum)
and `snr_per_pixel` when the variance method is used.

### 24.7 Spectrum Utilities

Implementations SHOULD provide the following utility functions:

| Function | Description |
|---|---|
| normalize_spectrum | Divide flux by a fitted continuum |
| resample_spectrum | Resample a spectrum onto a new wavelength grid, with optional variance propagation |
| smooth_spectrum | Smooth a spectrum with a boxcar or Gaussian kernel |
| equivalent_width | Compute rest-frame equivalent width over a specified line range |

---

## Appendix A: JSON-LD Context

The NOVA JSON-LD context defines the vocabulary mappings:

```json
{
  "@context": {
    "nova": "https://nova-astro.org/v0.1/vocab#",
    "prov": "http://www.w3.org/ns/prov#",
    "ivoa": "http://www.ivoa.net/rdf/vocab#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "nova:version": { "@type": "xsd:string" },
    "nova:created": { "@type": "xsd:dateTime" },
    "nova:naxes": { "@type": "xsd:integer" },
    "nova:crpix": { "@type": "xsd:double" },
    "nova:crval": { "@type": "xsd:double" },
    "nova:cd_matrix": { "@container": "@list" },
    "nova:sha256": { "@type": "xsd:string" }
  }
}
```

---

## Appendix B: Revision History

| Version | Date | Changes |
|---|---|---|
| 0.1-draft | 2026-04-12 | Initial specification draft |
| 0.2-draft | 2026-04-12 | Multi-extension support, table data, scaled integers, extended data types |
| 0.3-draft | 2026-04-12 | Remote access, batch migration, streaming I/O, pipeline adapters |
| 0.4-draft | 2026-04-12 | Image processing, photometry, spectral analysis, coordinate transforms, catalog operations |
| 0.5-draft | 2026-04-12 | Pipeline framework, native operations, astrometry/photometry/spectroscopy pipelines |
| 1.0-rc | 2026-04-12 | Release candidate consolidating all Phase 1-5 features |
