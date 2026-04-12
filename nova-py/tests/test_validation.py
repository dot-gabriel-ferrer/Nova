"""Tests for the NOVA validation module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nova.validation import (
    validate_metadata,
    validate_wcs,
    validate_provenance,
    validate_store,
)
from nova.container import create_dataset, NovaDataset
from nova.wcs import NovaWCS, WCSAxis, AffineTransform, CelestialFrame


class TestValidateMetadata:
    """Tests for metadata validation."""

    def test_valid_metadata(self) -> None:
        meta = {
            "@context": "https://nova-astro.org/v0.1/context.jsonld",
            "@type": "nova:Observation",
            "nova:version": "0.1.0",
            "nova:created": "2026-01-01T00:00:00Z",
        }
        errors = validate_metadata(meta)
        assert errors == []

    def test_missing_required_fields(self) -> None:
        errors = validate_metadata({})
        assert len(errors) == 4  # 4 required fields

    def test_invalid_context(self) -> None:
        meta = {
            "@context": "https://wrong.org/context.jsonld",
            "@type": "nova:Observation",
            "nova:version": "0.1.0",
            "nova:created": "2026-01-01T00:00:00Z",
        }
        errors = validate_metadata(meta)
        assert any("@context" in e for e in errors)

    def test_invalid_version(self) -> None:
        meta = {
            "@context": "https://nova-astro.org/v0.1/context.jsonld",
            "@type": "nova:Observation",
            "nova:version": "not-semver",
            "nova:created": "2026-01-01T00:00:00Z",
        }
        errors = validate_metadata(meta)
        assert any("version" in e for e in errors)

    def test_invalid_data_level(self) -> None:
        meta = {
            "@context": "https://nova-astro.org/v0.1/context.jsonld",
            "@type": "nova:Observation",
            "nova:version": "0.1.0",
            "nova:created": "2026-01-01T00:00:00Z",
            "nova:data_level": "L5",
        }
        errors = validate_metadata(meta)
        assert any("data_level" in e for e in errors)

    def test_valid_data_levels(self) -> None:
        for level in ("L0", "L1", "L2", "L3"):
            meta = {
                "@context": "https://nova-astro.org/v0.1/context.jsonld",
                "@type": "nova:Observation",
                "nova:version": "0.1.0",
                "nova:created": "2026-01-01T00:00:00Z",
                "nova:data_level": level,
            }
            errors = validate_metadata(meta)
            assert errors == [], f"Unexpected errors for {level}: {errors}"

    def test_valid_instrument(self) -> None:
        meta = {
            "@context": "https://nova-astro.org/v0.1/context.jsonld",
            "@type": "nova:Observation",
            "nova:version": "0.1.0",
            "nova:created": "2026-01-01T00:00:00Z",
            "nova:instrument": {
                "@type": "nova:Instrument",
                "nova:name": "DECam",
            },
        }
        errors = validate_metadata(meta)
        assert errors == []

    def test_invalid_instrument(self) -> None:
        meta = {
            "@context": "https://nova-astro.org/v0.1/context.jsonld",
            "@type": "nova:Observation",
            "nova:version": "0.1.0",
            "nova:created": "2026-01-01T00:00:00Z",
            "nova:instrument": {"@type": "wrong"},
        }
        errors = validate_metadata(meta)
        assert any("nova:Instrument" in e for e in errors)

    def test_context_as_array(self) -> None:
        meta = {
            "@context": [
                "https://nova-astro.org/v0.1/context.jsonld",
                "https://www.w3.org/ns/prov",
            ],
            "@type": "nova:Observation",
            "nova:version": "0.1.0",
            "nova:created": "2026-01-01T00:00:00Z",
        }
        errors = validate_metadata(meta)
        assert errors == []


class TestValidateWCS:
    """Tests for WCS validation."""

    def test_valid_wcs(self) -> None:
        wcs = NovaWCS(
            naxes=2,
            axes=[
                WCSAxis(0, "RA---TAN", 2048.0, 150.0, "deg"),
                WCSAxis(1, "DEC--TAN", 2048.0, 2.0, "deg"),
            ],
            transform=AffineTransform(
                cd_matrix=[[-7.3e-05, 0.0], [0.0, 7.3e-05]]
            ),
        )
        errors = validate_wcs(wcs.to_dict())
        assert errors == []

    def test_missing_fields(self) -> None:
        errors = validate_wcs({})
        assert len(errors) > 0

    def test_invalid_naxes(self) -> None:
        wcs_data = {
            "@context": "https://nova-astro.org/v0.1/context.jsonld",
            "@type": "nova:WCS",
            "nova:naxes": 10,
            "nova:axes": [],
            "nova:transform": {"@type": "nova:AffineTransform"},
        }
        errors = validate_wcs(wcs_data)
        assert any("naxes" in e for e in errors)

    def test_axes_count_mismatch(self) -> None:
        wcs_data = {
            "@context": "https://nova-astro.org/v0.1/context.jsonld",
            "@type": "nova:WCS",
            "nova:naxes": 2,
            "nova:axes": [{
                "@type": "nova:CelestialAxis",
                "nova:index": 0,
                "nova:ctype": "RA---TAN",
                "nova:crpix": 1024.0,
                "nova:crval": 150.0,
                "nova:unit": "deg",
            }],
            "nova:transform": {
                "@type": "nova:AffineTransform",
                "nova:cd_matrix": [[-1e-4, 0.0], [0.0, 1e-4]],
            },
        }
        errors = validate_wcs(wcs_data)
        assert any("length" in e for e in errors)


class TestValidateProvenance:
    """Tests for provenance validation."""

    def test_valid_provenance(self) -> None:
        prov = {
            "@context": [
                "https://nova-astro.org/v0.1/context.jsonld",
                "https://www.w3.org/ns/prov",
            ],
            "@type": "prov:Bundle",
            "prov:entity": [
                {"@id": "nova:entity/raw-001", "@type": "prov:Entity"},
            ],
            "prov:activity": [
                {
                    "@id": "nova:activity/cal",
                    "@type": "prov:Activity",
                    "prov:used": {"@id": "nova:entity/raw-001"},
                },
            ],
        }
        errors = validate_provenance(prov, data_level="L1")
        assert errors == []

    def test_l1_requires_provenance(self) -> None:
        prov = {
            "@context": ["https://nova-astro.org/v0.1/context.jsonld"],
            "@type": "prov:Bundle",
        }
        errors = validate_provenance(prov, data_level="L1")
        assert any("INV-5" in e for e in errors)

    def test_l0_no_provenance_required(self) -> None:
        prov = {
            "@context": ["https://nova-astro.org/v0.1/context.jsonld"],
            "@type": "prov:Bundle",
        }
        errors = validate_provenance(prov, data_level="L0")
        assert errors == []

    def test_unknown_entity_reference(self) -> None:
        prov = {
            "@context": ["https://nova-astro.org/v0.1/context.jsonld"],
            "@type": "prov:Bundle",
            "prov:entity": [],
            "prov:activity": [
                {
                    "@id": "nova:activity/test",
                    "@type": "prov:Activity",
                    "prov:used": {"@id": "nova:entity/nonexistent"},
                },
            ],
        }
        errors = validate_provenance(prov)
        assert any("nonexistent" in e for e in errors)


class TestValidateStore:
    """Tests for full store validation."""

    def test_validate_valid_store(self, tmp_path: Path) -> None:
        store_path = tmp_path / "valid.nova.zarr"
        data = np.ones((64, 64))
        wcs = NovaWCS(
            naxes=2,
            axes=[
                WCSAxis(0, "RA---TAN", 32.0, 150.0, "deg"),
                WCSAxis(1, "DEC--TAN", 32.0, 2.0, "deg"),
            ],
            transform=AffineTransform(
                cd_matrix=[[-1e-4, 0.0], [0.0, 1e-4]]
            ),
        )
        create_dataset(store_path, data, wcs=wcs)

        results = validate_store(store_path)
        for filename, errors in results.items():
            assert errors == [], f"{filename}: {errors}"

    def test_validate_nonexistent_store(self) -> None:
        results = validate_store("/nonexistent/path")
        assert "store" in results
        assert len(results["store"]) > 0

    def test_validate_missing_metadata(self, tmp_path: Path) -> None:
        store_path = tmp_path / "incomplete.nova.zarr"
        store_path.mkdir()
        results = validate_store(store_path)
        assert len(results["nova_metadata.json"]) > 0
