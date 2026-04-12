"""Tutorial 04: W3C PROV-DM Provenance -- Tracking data lineage.

This tutorial demonstrates NOVA's provenance tracking:
  1. Create a raw observation (L0 data) -- no provenance required
  2. Create a calibrated dataset (L1) -- provenance REQUIRED (INV-5)
  3. Build a processing pipeline provenance chain
  4. Validate provenance requirements
  5. Inspect and query the provenance graph

Run:
    cd nova-py
    python tutorials/04_provenance.py
"""

from __future__ import annotations

import json
import datetime
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    """Run the provenance tutorial."""
    from nova.container import NovaDataset
    from nova.provenance import (
        ProvenanceBundle,
        ProvenanceEntity,
        ProvenanceActivity,
        ProvenanceAgent,
    )

    print("=" * 70)
    print("  Provenance Tracking Tutorial")
    print("  W3C PROV-DM data lineage with NOVA")
    print("=" * 70)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # -- Step 1: L0 Raw Data -- No provenance required -----------------
        print("Step 1: Create L0 raw data (provenance optional)")
        print("-" * 70)

        rng = np.random.default_rng(42)
        raw_data = rng.normal(1000.0, 50.0, (512, 512)).astype(np.float32)

        raw_path = tmpdir / "raw_observation.nova.zarr"
        ds_raw = NovaDataset(raw_path, mode="w")
        ds_raw.set_science_data(raw_data)
        ds_raw._metadata["nova:data_level"] = "L0"
        ds_raw.save()
        ds_raw.close()

        # Validate -- L0 doesn't require provenance
        empty_prov = ProvenanceBundle()
        errors = empty_prov.validate("L0")
        print(f"  Data level:    L0 (raw)")
        print(f"  Has provenance: No")
        print(f"  Validation:    {'OK PASS' if not errors else 'FAIL FAIL'}")
        print(f"  -> L0 data does NOT require provenance (INV-5)")
        print()

        # -- Step 2: L1 Calibrated Data -- Provenance REQUIRED -------------
        print("Step 2: Create L1 calibrated data (provenance REQUIRED)")
        print("-" * 70)

        # Simulate calibration: bias subtraction + flat-field
        bias = np.full((512, 512), 100.0, dtype=np.float32)
        flat = rng.uniform(0.95, 1.05, (512, 512)).astype(np.float32)
        calibrated = (raw_data - bias) / flat

        # First, show that L1 without provenance FAILS validation
        no_prov = ProvenanceBundle()
        errors = no_prov.validate("L1")
        print(f"  Without provenance:")
        print(f"    Validation: FAIL FAIL")
        for err in errors:
            print(f"    Error: {err}")
        print()

        # Now create proper provenance
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        prov = ProvenanceBundle()

        # Entities: raw input, bias frame, flat field, calibrated output
        prov.add_entity(ProvenanceEntity(
            entity_id="nova:entity/raw_observation",
            data_level="L0",
            filename="raw_observation.nova.zarr",
            data_format="NOVA",
        ))
        prov.add_entity(ProvenanceEntity(
            entity_id="nova:entity/master_bias",
            data_level="L0",
            filename="master_bias.nova.zarr",
            data_format="NOVA",
        ))
        prov.add_entity(ProvenanceEntity(
            entity_id="nova:entity/master_flat",
            data_level="L0",
            filename="master_flat.nova.zarr",
            data_format="NOVA",
        ))
        prov.add_entity(ProvenanceEntity(
            entity_id="nova:entity/calibrated_observation",
            data_level="L1",
            filename="calibrated.nova.zarr",
            data_format="NOVA",
            derived_from="nova:entity/raw_observation",
            generated_by="nova:activity/calibration",
        ))

        # Activity: the calibration process
        prov.add_activity(ProvenanceActivity(
            activity_id="nova:activity/calibration",
            started_at=now,
            ended_at=now,
            used=[
                "nova:entity/raw_observation",
                "nova:entity/master_bias",
                "nova:entity/master_flat",
            ],
            generated=["nova:entity/calibrated_observation"],
            software="nova-py v0.1.0",
            algorithm="bias_subtraction_flat_field",
            parameters={
                "bias_method": "master_bias_subtraction",
                "flat_method": "normalized_flat_division",
            },
        ))

        # Agent: the software
        prov.add_agent(ProvenanceAgent(
            agent_id="nova:agent/nova-py",
            name="nova-py",
            version="0.1.0",
        ))

        # Validate
        errors = prov.validate("L1")
        print(f"  With provenance:")
        print(f"    Entities:   {len(prov.entities)}")
        print(f"    Activities: {len(prov.activities)}")
        print(f"    Agents:     {len(prov.agents)}")
        print(f"    Validation: {'OK PASS' if not errors else 'FAIL FAIL'}")
        print()

        # Save calibrated data with provenance
        cal_path = tmpdir / "calibrated.nova.zarr"
        ds_cal = NovaDataset(cal_path, mode="w")
        ds_cal.set_science_data(calibrated)
        ds_cal._metadata["nova:data_level"] = "L1"
        ds_cal.provenance = prov
        ds_cal.save()
        ds_cal.close()

        # -- Step 3: Inspect the provenance chain -------------------------
        print("Step 3: Inspect the provenance JSON-LD")
        print("-" * 70)

        prov_path = cal_path / "provenance.json"
        with open(prov_path) as f:
            prov_json = json.load(f)

        print("  Provenance Bundle (JSON-LD):")
        print(json.dumps(prov_json, indent=2)[:1200])
        print("  ...")
        print()

        # -- Step 4: Build a multi-step pipeline --------------------------
        print("Step 4: Multi-step pipeline provenance (L0 -> L1 -> L2)")
        print("-" * 70)

        # Add L2 processing (e.g., source extraction)
        prov_l2 = ProvenanceBundle()

        # Carry over L1 entities
        for e in prov.entities:
            prov_l2.add_entity(e)

        # Add L2 output
        prov_l2.add_entity(ProvenanceEntity(
            entity_id="nova:entity/source_catalog",
            data_level="L2",
            filename="source_catalog.nova.zarr",
            data_format="NOVA",
            derived_from="nova:entity/calibrated_observation",
            generated_by="nova:activity/source_extraction",
        ))

        # Carry over L1 activity
        for a in prov.activities:
            prov_l2.add_activity(a)

        # Add L2 activity
        prov_l2.add_activity(ProvenanceActivity(
            activity_id="nova:activity/source_extraction",
            started_at=now,
            ended_at=now,
            used=["nova:entity/calibrated_observation"],
            generated=["nova:entity/source_catalog"],
            software="nova-py v0.1.0",
            algorithm="daofind_source_extraction",
            parameters={
                "threshold_sigma": 5.0,
                "fwhm_pixels": 3.5,
            },
        ))

        prov_l2.add_agent(ProvenanceAgent(
            agent_id="nova:agent/nova-py",
            name="nova-py",
            version="0.1.0",
        ))

        # Validate L2
        errors = prov_l2.validate("L2")
        print(f"  Pipeline chain: L0 -> L1 (calibration) -> L2 (source extraction)")
        print(f"  Total entities:   {len(prov_l2.entities)}")
        print(f"  Total activities: {len(prov_l2.activities)}")
        print(f"  Validation:       {'OK PASS' if not errors else 'FAIL FAIL'}")
        print()

        # -- Step 5: Query the provenance graph ----------------------------
        print("Step 5: Query the provenance graph")
        print("-" * 70)

        prov_dict = prov_l2.to_dict()

        # Find what generated the source catalog
        for entity in prov_dict["prov:entity"]:
            if entity["@id"] == "nova:entity/source_catalog":
                gen_by = entity.get("prov:wasGeneratedBy", {})
                derived = entity.get("prov:wasDerivedFrom", {})
                print(f"  Source catalog:")
                print(f"    Generated by: {gen_by.get('@id', 'unknown')}")
                print(f"    Derived from: {derived.get('@id', 'unknown')}")

        # Find what the calibration used
        for activity in prov_dict["prov:activity"]:
            if activity["@id"] == "nova:activity/calibration":
                used = activity.get("prov:used", [])
                if isinstance(used, list):
                    inputs = [u["@id"] for u in used]
                else:
                    inputs = [used["@id"]]
                print(f"  Calibration inputs: {inputs}")
                print(f"  Algorithm: {activity.get('nova:algorithm', 'N/A')}")
                params = activity.get("nova:parameters", {})
                for key, val in params.items():
                    print(f"    {key}: {val}")

        print()
        print("  -> Complete provenance chain is machine-readable and queryable!")
        print("  -> Meets W3C PROV-DM standard for interoperability.")

    print()
    print("=" * 70)
    print("  OK Tutorial complete! Provenance tracking demonstrated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
