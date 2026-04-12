# NOVA Format Specification — Version 0.1 (Draft)

**Status:** Working Draft  
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

---

## 2. Design Invariants

All implementations MUST satisfy these invariants. No specification revision
SHALL violate them without unanimous Working Group approval.

| ID | Invariant | Requirement |
|---|---|---|
| INV-1 | BACKWARD_COMPAT | Lossless FITS↔NOVA conversion for all valid FITS files |
| INV-2 | CLOUD_FIRST | Chunk index MUST reside in first 8KB; any region accessible in ≤2 HTTP requests |
| INV-3 | HUMAN_READABLE | JSON-LD metadata manifest MUST be readable with any text editor |
| INV-4 | INTEGRITY_BY_DEFAULT | Every chunk MUST have a SHA-256 hash in the index |
| INV-5 | PROV_MANDATORY | W3C PROV-DM provenance REQUIRED for reduced/calibrated data |
| INV-6 | PARALLEL_WRITE | Format MUST support concurrent writes without global locks |
| INV-7 | ML_NATIVE | Native float16/BFloat16 support with standardized normalization metadata |

---

## 3. Layer 1 — Container & Access

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
3. Include SHA-256 hashes for each chunk (see §6)

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

1. **Request 1:** Fetch `nova_index.json` (≤8KB)
2. **Request 2:** Fetch the specific chunk(s) using byte range from the index

### 3.4 File Extension

NOVA files MUST use one of the following extensions:
- `.nova.zarr` — Directory store
- `.nova.zip` — ZIP-archived store

### 3.5 Magic Bytes

When stored as a single-file archive, the first 8 bytes MUST be:
```
\x89NOVA\r\n\x1a\n
```
(Inspired by the PNG signature for file type detection and corruption detection.)

---

## 4. Layer 2 — Compression & Encoding

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

## 5. Layer 3 — Scientific Data

### 5.1 Array Groups

A NOVA store organizes data into Zarr groups:

```
observation.nova.zarr/
├── nova_index.json          # Chunk index (§3.2)
├── nova_metadata.json       # Root metadata (§7)
├── data/
│   ├── science/             # Primary science array
│   │   ├── zarr.json
│   │   └── c.<chunk_keys>
│   ├── uncertainty/         # Co-located uncertainty plane
│   │   ├── zarr.json
│   │   └── c.<chunk_keys>
│   ├── mask/                # Data quality mask
│   │   ├── zarr.json
│   │   └── c.<chunk_keys>
│   └── preview/             # JPEG-XL preview (optional)
│       ├── zarr.json
│       └── c.<chunk_keys>
└── wcs.json                 # WCS metadata (§8)
```

### 5.2 Science Array

The primary science data MUST be stored in `data/science/` as a Zarr v3 array.

- The array dtype MUST be one of the types in §4.3.
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

---

## 6. Layer 4 — Integrity

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

## 7. Layer 4 — Metadata

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

## 8. Layer 5 — World Coordinate System (WCS)

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

The FITS converter (§10) MUST map all standard FITS WCS keywords to the
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

## 9. Layer 5 — Provenance

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

A conforming NOVA implementation MUST provide a FITS→NOVA converter that:

1. Preserves all header keywords as JSON-LD metadata
2. Converts WCS keywords to the structured WCS schema (§8)
3. Converts BITPIX-encoded data to the appropriate Zarr dtype
4. Preserves FITS extensions as separate Zarr groups
5. Generates SHA-256 hashes for all data chunks
6. Stores the original FITS header verbatim in `fits_origin/header.txt`

### 10.2 NOVA to FITS Conversion

A conforming NOVA implementation MUST provide a NOVA→FITS converter that:

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
