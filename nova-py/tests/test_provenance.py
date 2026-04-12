"""Tests for the NOVA provenance module."""

from __future__ import annotations

import json

from nova.provenance import (
    ProvenanceBundle,
    ProvenanceEntity,
    ProvenanceActivity,
    ProvenanceAgent,
)


class TestProvenanceEntity:
    """Tests for ProvenanceEntity."""

    def test_roundtrip(self) -> None:
        entity = ProvenanceEntity(
            entity_id="nova:entity/test-001",
            data_level="L0",
            filename="test.fits",
            data_format="FITS",
        )
        d = entity.to_dict()
        entity2 = ProvenanceEntity.from_dict(d)
        assert entity2.entity_id == entity.entity_id
        assert entity2.data_level == entity.data_level
        assert entity2.filename == entity.filename

    def test_with_derivation(self) -> None:
        entity = ProvenanceEntity(
            entity_id="nova:entity/calibrated-001",
            data_level="L1",
            derived_from="nova:entity/raw-001",
            generated_by="nova:activity/calibration",
        )
        d = entity.to_dict()
        assert d["prov:wasDerivedFrom"]["@id"] == "nova:entity/raw-001"
        assert d["prov:wasGeneratedBy"]["@id"] == "nova:activity/calibration"


class TestProvenanceActivity:
    """Tests for ProvenanceActivity."""

    def test_roundtrip(self) -> None:
        activity = ProvenanceActivity(
            activity_id="nova:activity/bias-subtraction",
            software="nova-py v0.1.0",
            algorithm="median_combine",
            parameters={"sigma_clip": 3.0},
            used=["nova:entity/raw-001"],
            generated=["nova:entity/calibrated-001"],
        )
        d = activity.to_dict()
        activity2 = ProvenanceActivity.from_dict(d)
        assert activity2.activity_id == activity.activity_id
        assert activity2.software == "nova-py v0.1.0"
        assert activity2.parameters == {"sigma_clip": 3.0}
        assert activity2.used == ["nova:entity/raw-001"]


class TestProvenanceBundle:
    """Tests for ProvenanceBundle."""

    def test_empty_bundle(self) -> None:
        bundle = ProvenanceBundle()
        d = bundle.to_dict()
        assert d["@type"] == "prov:Bundle"
        assert "@context" in d

    def test_full_bundle_roundtrip(self) -> None:
        bundle = ProvenanceBundle()
        bundle.add_entity(ProvenanceEntity(
            entity_id="nova:entity/raw-001",
            data_level="L0",
        ))
        bundle.add_entity(ProvenanceEntity(
            entity_id="nova:entity/cal-001",
            data_level="L1",
            derived_from="nova:entity/raw-001",
        ))
        bundle.add_activity(ProvenanceActivity(
            activity_id="nova:activity/calibrate",
            software="nova-py v0.1.0",
            used=["nova:entity/raw-001"],
            generated=["nova:entity/cal-001"],
        ))
        bundle.add_agent(ProvenanceAgent(
            agent_id="nova:agent/nova-py",
            name="nova-py",
            version="0.1.0",
        ))

        d = bundle.to_dict()
        json_str = json.dumps(d, indent=2)
        parsed = json.loads(json_str)
        bundle2 = ProvenanceBundle.from_dict(parsed)

        assert len(bundle2.entities) == 2
        assert len(bundle2.activities) == 1
        assert len(bundle2.agents) == 1

    def test_validate_l1_requires_provenance(self) -> None:
        """L1+ data must have provenance (INV-5)."""
        bundle = ProvenanceBundle()  # Empty
        errors = bundle.validate("L1")
        assert len(errors) == 2  # Missing entity and activity

    def test_validate_l0_no_provenance_required(self) -> None:
        """L0 data does not require provenance."""
        bundle = ProvenanceBundle()  # Empty
        errors = bundle.validate("L0")
        assert len(errors) == 0

    def test_validate_references(self) -> None:
        """Activities must reference known entities."""
        bundle = ProvenanceBundle()
        bundle.add_activity(ProvenanceActivity(
            activity_id="nova:activity/test",
            used=["nova:entity/nonexistent"],
        ))
        errors = bundle.validate("L0")
        assert any("nonexistent" in e for e in errors)

    def test_json_ld_structure(self) -> None:
        """Verify JSON-LD context and type."""
        bundle = ProvenanceBundle()
        d = bundle.to_dict()
        assert "https://nova-astro.org/v0.1/context.jsonld" in d["@context"]
        assert "https://www.w3.org/ns/prov" in d["@context"]
        assert d["@type"] == "prov:Bundle"
