"""NOVA metadata validation module.

Validates NOVA metadata files against the JSON Schemas defined in the spec.
Supports validation of nova_metadata.json, wcs.json, and provenance.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Path to schemas directory (relative to the repo root)
_SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "spec" / "schemas"


def _load_schema(schema_name: str) -> dict[str, Any]:
    """Load a JSON Schema from the spec/schemas directory.

    Parameters
    ----------
    schema_name : str
        Schema filename (e.g., 'nova-metadata.schema.json').

    Returns
    -------
    dict
        Parsed JSON Schema.

    Raises
    ------
    FileNotFoundError
        If the schema file does not exist.
    """
    schema_path = _SCHEMA_DIR / schema_name
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with open(schema_path) as f:
        return json.load(f)


def validate_metadata(data: dict[str, Any]) -> list[str]:
    """Validate a NOVA metadata document against the schema.

    Performs structural validation without requiring jsonschema library.
    Checks required fields, types, and NOVA-specific constraints.

    Parameters
    ----------
    data : dict
        Metadata dictionary (contents of nova_metadata.json).

    Returns
    -------
    list of str
        Validation errors. Empty list means valid.
    """
    errors: list[str] = []

    # Required fields
    required = ["@context", "@type", "nova:version", "nova:created"]
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    # Validate @context
    ctx = data.get("@context")
    if ctx is not None:
        valid_ctx = "https://nova-astro.org/v0.1/context.jsonld"
        if isinstance(ctx, str):
            if ctx != valid_ctx:
                errors.append(
                    f"Invalid @context: expected '{valid_ctx}', got '{ctx}'"
                )
        elif isinstance(ctx, list):
            if valid_ctx not in ctx:
                errors.append(
                    f"@context array must contain '{valid_ctx}'"
                )
        else:
            errors.append("@context must be a string or array")

    # Validate version format (semver)
    version = data.get("nova:version", "")
    if version and not _is_semver(version):
        errors.append(
            f"Invalid version format: '{version}'. Expected semver (e.g., '0.1.0')."
        )

    # Validate data level
    data_level = data.get("nova:data_level")
    if data_level is not None and data_level not in ("L0", "L1", "L2", "L3"):
        errors.append(
            f"Invalid data_level: '{data_level}'. Must be L0, L1, L2, or L3."
        )

    # Validate instrument structure
    instrument = data.get("nova:instrument")
    if instrument is not None:
        if not isinstance(instrument, dict):
            errors.append("nova:instrument must be an object")
        else:
            if instrument.get("@type") != "nova:Instrument":
                errors.append(
                    "nova:instrument must have @type 'nova:Instrument'"
                )
            if "nova:name" not in instrument:
                errors.append("nova:instrument must have 'nova:name'")

    return errors


def validate_wcs(data: dict[str, Any]) -> list[str]:
    """Validate a NOVA WCS document against the schema.

    Parameters
    ----------
    data : dict
        WCS dictionary (contents of wcs.json).

    Returns
    -------
    list of str
        Validation errors. Empty list means valid.
    """
    errors: list[str] = []

    # Required fields
    required = ["@context", "@type", "nova:naxes", "nova:axes", "nova:transform"]
    for field in required:
        if field not in data:
            errors.append(f"Missing required WCS field: '{field}'")

    # Validate @type
    if data.get("@type") != "nova:WCS":
        errors.append("WCS @type must be 'nova:WCS'")

    # Validate naxes
    naxes = data.get("nova:naxes")
    if naxes is not None:
        if not isinstance(naxes, int) or naxes < 1 or naxes > 7:
            errors.append(
                f"nova:naxes must be an integer between 1 and 7, got {naxes}"
            )

    # Validate axes
    axes = data.get("nova:axes", [])
    if isinstance(axes, list):
        if naxes is not None and len(axes) != naxes:
            errors.append(
                f"nova:axes length ({len(axes)}) must equal nova:naxes ({naxes})"
            )

        valid_axis_types = {
            "nova:CelestialAxis",
            "nova:SpectralAxis",
            "nova:TemporalAxis",
            "nova:StokesAxis",
            "nova:LinearAxis",
        }
        for i, axis in enumerate(axes):
            if not isinstance(axis, dict):
                errors.append(f"Axis {i} must be an object")
                continue
            axis_type = axis.get("@type")
            if axis_type and axis_type not in valid_axis_types:
                errors.append(
                    f"Axis {i} has invalid @type: '{axis_type}'"
                )
            for req_field in ("nova:index", "nova:ctype", "nova:crpix",
                              "nova:crval", "nova:unit"):
                if req_field not in axis:
                    errors.append(
                        f"Axis {i} missing required field: '{req_field}'"
                    )

    # Validate transform
    transform = data.get("nova:transform")
    if transform is not None:
        if not isinstance(transform, dict):
            errors.append("nova:transform must be an object")
        else:
            if "@type" not in transform:
                errors.append("nova:transform must have @type")
            if transform.get("@type") == "nova:AffineTransform":
                cd = transform.get("nova:cd_matrix")
                if cd is None:
                    errors.append(
                        "AffineTransform must have nova:cd_matrix"
                    )
                elif isinstance(cd, list) and naxes is not None:
                    if len(cd) != naxes:
                        errors.append(
                            f"CD matrix rows ({len(cd)}) must equal naxes ({naxes})"
                        )

    return errors


def validate_provenance(
    data: dict[str, Any],
    data_level: str = "L0",
) -> list[str]:
    """Validate a NOVA provenance document against the schema.

    Parameters
    ----------
    data : dict
        Provenance dictionary (contents of provenance.json).
    data_level : str
        Data level to check mandatory provenance (INV-5).

    Returns
    -------
    list of str
        Validation errors. Empty list means valid.
    """
    errors: list[str] = []

    # Validate @type
    if data.get("@type") != "prov:Bundle":
        errors.append("Provenance @type must be 'prov:Bundle'")

    # Validate @context includes both NOVA and PROV
    ctx = data.get("@context")
    if ctx is not None:
        nova_ctx = "https://nova-astro.org/v0.1/context.jsonld"
        if isinstance(ctx, list):
            if nova_ctx not in ctx:
                errors.append(
                    f"Provenance @context must include '{nova_ctx}'"
                )
        elif isinstance(ctx, str):
            if ctx != nova_ctx:
                errors.append(
                    f"Provenance @context must include '{nova_ctx}'"
                )

    # Validate entities
    entities = data.get("prov:entity", [])
    entity_ids: set[str] = set()
    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            errors.append(f"Entity {i} must be an object")
            continue
        eid = entity.get("@id")
        if not eid:
            errors.append(f"Entity {i} missing @id")
        else:
            entity_ids.add(eid)

    # Validate activities reference known entities
    activities = data.get("prov:activity", [])
    for i, activity in enumerate(activities):
        if not isinstance(activity, dict):
            errors.append(f"Activity {i} must be an object")
            continue
        if not activity.get("@id"):
            errors.append(f"Activity {i} missing @id")

        # Check used references
        used = activity.get("prov:used")
        if used is not None:
            refs = _extract_refs(used)
            for ref in refs:
                if ref not in entity_ids:
                    errors.append(
                        f"Activity {i} references unknown entity: '{ref}'"
                    )

    # INV-5: L1+ data requires provenance
    if data_level in ("L1", "L2", "L3"):
        if not entities:
            errors.append(
                f"Data level {data_level} requires at least one "
                f"provenance entity (INV-5: PROV_MANDATORY)"
            )
        if not activities:
            errors.append(
                f"Data level {data_level} requires at least one "
                f"provenance activity (INV-5: PROV_MANDATORY)"
            )

    return errors


def validate_store(store_path: str | Path) -> dict[str, list[str]]:
    """Validate an entire NOVA store.

    Validates all metadata files in the store directory.

    Parameters
    ----------
    store_path : str or Path
        Path to the NOVA store directory.

    Returns
    -------
    dict
        Dictionary mapping file names to lists of validation errors.
        Empty lists indicate valid files.
    """
    store_path = Path(store_path)
    results: dict[str, list[str]] = {}

    if not store_path.exists():
        return {"store": [f"Store not found: {store_path}"]}

    # Validate nova_metadata.json
    meta_path = store_path / "nova_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        results["nova_metadata.json"] = validate_metadata(meta)
        data_level = meta.get("nova:data_level", "L0")
    else:
        results["nova_metadata.json"] = ["File not found: nova_metadata.json"]
        data_level = "L0"

    # Validate wcs.json
    wcs_path = store_path / "wcs.json"
    if wcs_path.exists():
        with open(wcs_path) as f:
            wcs = json.load(f)
        results["wcs.json"] = validate_wcs(wcs)
    else:
        results["wcs.json"] = []  # WCS is optional

    # Validate provenance.json
    prov_path = store_path / "provenance.json"
    if prov_path.exists():
        with open(prov_path) as f:
            prov = json.load(f)
        results["provenance.json"] = validate_provenance(prov, data_level)
    else:
        if data_level in ("L1", "L2", "L3"):
            results["provenance.json"] = [
                f"provenance.json required for data level {data_level} "
                f"(INV-5: PROV_MANDATORY)"
            ]
        else:
            results["provenance.json"] = []

    # Validate nova_index.json (optional — only built on demand)
    index_path = store_path / "nova_index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        results["nova_index.json"] = _validate_index(index)
    else:
        results["nova_index.json"] = []  # Index is optional (built on demand)

    return results


def _validate_index(data: dict[str, Any]) -> list[str]:
    """Validate a NOVA chunk index."""
    errors: list[str] = []

    if data.get("@type") != "nova:ChunkIndex":
        errors.append("Chunk index @type must be 'nova:ChunkIndex'")

    chunks = data.get("nova:chunks", [])
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            errors.append(f"Chunk {i} must be an object")
            continue
        if "nova:path" not in chunk:
            errors.append(f"Chunk {i} missing nova:path")
        if "nova:sha256" not in chunk:
            errors.append(f"Chunk {i} missing nova:sha256 (INV-4)")
        if "nova:size" not in chunk:
            errors.append(f"Chunk {i} missing nova:size")

    return errors


def _extract_refs(value: Any) -> list[str]:
    """Extract @id references from a JSON-LD value."""
    refs: list[str] = []
    if isinstance(value, dict):
        if "@id" in value:
            refs.append(value["@id"])
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "@id" in item:
                refs.append(item["@id"])
    return refs


def _is_semver(version: str) -> bool:
    """Check if a string follows semver format."""
    parts = version.split(".")
    if len(parts) != 3:
        return False
    return all(part.isdigit() for part in parts)
