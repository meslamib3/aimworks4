# FAIR / Provenance Template Pack

This folder contains 10 FAIR/provenance templates for the AIMWORKS template-driven builder.

All templates use the fields accepted by `TemplateSpec`:
- id, label, profile, adds, relationships, curated_values, units_map, shapes_embed

They purposely avoid a `description` field to keep compatibility with your loader.

## Files
- FAIR_DatasetDescriptor.json
- FAIR_License_Access.json
- FAIR_Creators_ORCID.json
- FAIR_Instrument_Config.json
- FAIR_Calibration_Records.json
- FAIR_Environment_Log.json
- FAIR_RunLog_Deviation.json
- FAIR_Sample_Manifest.json
- FAIR_Processing_Code.json
- FAIR_Validation_Report.json

## Notes
- Embedded SHACL shapes check DCAT/DCTerms minima and structural links (`isPartOf`, `hasPart`).
- Where numeric context is useful (environment, run log, sample counts), units are included.
- Units are mapped to QUDT tokens via `units_map` so your builder can resolve them.
