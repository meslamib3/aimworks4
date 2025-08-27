# streamlit_app7.py
# Run with: streamlit run streamlit_app7.py
import os
import re
import json
from typing import Dict, List, Tuple, Optional

import streamlit as st
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Allow running without openai installed for non-LLM paths

from ontology_core import (
    BASE, HEADINGS, ALLOWED_PREDICATES, clean_text, to_camel_case, to_valid_id,
    build_graph, run_shacl_validation, load_unit_aliases
)
from template_lib import (
    TemplateSpec, load_templates_from_dir, rank_templates, apply_template
)

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="AIMWORKS - Template-Driven Ontology Builder", layout="wide")
st.title("AIMWORKS - Template-Driven Ontology Builder")

def do_rerun():
    try:
        st.rerun()
    except Exception:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.subheader("LLM (optional for classification)")
    api_key = st.text_input("OpenAI API Key", type="password")
    model_name = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], index=0)
    temperature = st.slider("Temperature", 0.0, 0.8, 0.2, 0.05)

    st.subheader("Validation & Export")
    run_shacl = st.checkbox("Run SHACL validation", value=True)
    block_on_shacl = st.checkbox("Block export if SHACL fails", value=True)

    # NEW: A9R options
    st.markdown("---")
    st.subheader("Auto-repair (A9R)")
    auto_repair_on_fail = st.checkbox("Auto-repair if SHACL fails (A9R)", value=True)
    max_repair_passes = st.number_input("Max auto-repair passes", min_value=1, max_value=10, value=2, step=1)

    st.subheader("Session controls")
    if st.button("Reload templates and reset plan"):
        for k in [
            "TEMPLATES","classification","triples","curated_values","datapoints",
            "template_shapes","unit_aliases","tpl_select_labels","tpl_select_ids",
            "filter_text","dataset_meta","seed_features_txt","prev_question","question"
        ]:
            st.session_state.pop(k, None)
        do_rerun()

# Client is optional (only for "Classify with LLM")
client = None
if api_key and OpenAI is not None:
    client = OpenAI(api_key=api_key.strip())

# ----------------------------
# Template library (as-is)
# ----------------------------
if "TEMPLATES" not in st.session_state:
    TEMPLATES: List[TemplateSpec] = load_templates_from_dir("templates")
    st.session_state["TEMPLATES"] = TEMPLATES
else:
    TEMPLATES = st.session_state["TEMPLATES"]

if not TEMPLATES:
    st.error("No templates found under ./templates (recursively).")
    st.stop()

# Category counts (sanity)
cat_counts: Dict[str, int] = {"measurement": 0, "metric": 0, "context": 0, "instrument": 0, "process": 0, "fair": 0, "other": 0}
for t in TEMPLATES:
    path = (getattr(t, "src_path", "") or "").replace("\\", "/")
    if "/measurement/" in path:
        cat_counts["measurement"] += 1
    elif "/metric/" in path:
        cat_counts["metric"] += 1
    elif "/context/" in path:
        cat_counts["context"] += 1
    elif "/instrument/" in path:
        cat_counts["instrument"] += 1
    elif "/process/" in path:
        cat_counts["process"] += 1
    elif "/fair/" in path:
        cat_counts["fair"] += 1
    else:
        cat_counts["other"] += 1

st.success(
    f"{len(TEMPLATES)} templates loaded  -  "
    f"M:{cat_counts['measurement']}  Met:{cat_counts['metric']}  Ctx:{cat_counts['context']}  "
    f"Inst:{cat_counts['instrument']}  Proc:{cat_counts['process']}  FAIR:{cat_counts['fair']}  Other:{cat_counts['other']}"
)

# ----------------------------
# Helpers (as-is)
# ----------------------------
def llm_chat(sys_prompt: str, user_prompt: str, max_tokens: int = 1000, json_mode=False) -> str:
    if not client:
        return "{}"
    messages = [{"role":"system","content": sys_prompt}, {"role":"user","content": user_prompt}]
    if json_mode:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type":"json_object"}
        )
    else:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    return (resp.choices[0].message.content or "").strip()

def extract_json_object(text: str):
    if not text: return None
    t = text.strip().replace("\u200b","").replace("\ufeff","")
    m = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", t, re.I)
    if m: return m.group(1).strip()
    start = t.find("{")
    if start == -1: return None
    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i+1].strip()
    return None

def split_top_level_commas(line: str) -> List[str]:
    parts, buf, depth = [], "", 0
    for ch in line:
        if ch == "(": depth += 1
        elif ch == ")": depth = max(0, depth-1)
        if ch == "," and depth == 0:
            if buf.strip():
                parts.append(buf.strip()); buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts

# ----------------------------
# Inputs (as-is)
# ----------------------------
st.markdown("### Step 1 - Research question and seed features")
default_q = "Evaluate IrOx/SrTiO3-x OER catalysts by Polarization Curve and EIS in 0.5 M H2SO4 at 25 C; report Tafel slope, overpotential at 10 mA/cm2, and Rct; include a 10 h constant-current durability hold."
question = st.text_input("Research Question", value=st.session_state.get("question", default_q))
prev_q = st.session_state.get("prev_question")
st.session_state["question"] = question

# Auto-reset selection if question changed
if prev_q is None:
    st.session_state["prev_question"] = question
elif prev_q != question:
    for k in ["tpl_select_labels","tpl_select_ids"]:
        st.session_state.pop(k, None)
    st.session_state["prev_question"] = question

seed_default = "PolarizationCurve, EIS, CurrentDensity, TafelSlope, Overpotential, ChargeTransferResistance, ReferenceScale (RHE), IRCompensationFraction (0.85), Duration (10 h), ElectrolyteConcentration (0.5 M), Temperature (25 C)"
seed_features_txt = st.text_area("Seed features (comma-separated)", value=st.session_state.get("seed_features_txt", seed_default), height=90)
st.session_state["seed_features_txt"] = seed_features_txt

features_raw = [f.strip() for f in split_top_level_commas(seed_features_txt) if f.strip()]
feature_contexts: Dict[str,str] = {}
features: List[str] = []
for f in features_raw:
    if "(" in f and ")" in f:
        base = to_camel_case(f.split("(",1)[0].strip())
        ctx  = f.split("(",1)[1].rstrip(")").strip()
        features.append(base); feature_contexts[base] = ctx
    else:
        features.append(to_camel_case(f))
features = list(dict.fromkeys(features))
st.caption("Parsed features: " + (", ".join(features) if features else "—"))

# ----------------------------
# Classification (optional, as-is)
# ----------------------------
st.markdown("### Step 2 - Classify features (optional)")
if st.button("Classify with LLM"):
    sys_p = "You classify items into one of these ontology classes: " + ", ".join(HEADINGS)
    user_p = "Return JSON with exactly those keys; values are arrays. Items:\n" + json.dumps(features, indent=2)
    raw = llm_chat(sys_p, user_p, json_mode=True, max_tokens=1200)
    txt = extract_json_object(raw) or raw
    try:
        classification = json.loads(txt)
    except Exception:
        classification = {h:[] for h in HEADINGS}
    # Guard: common methods must end in Measurement
    for m in ["PolarizationCurve","EIS","ElectrochemicalImpedanceSpectroscopy","RRDE","RDE","ORR","OER","HER"]:
        if m in features and m not in (classification.get("Measurement") or []):
            classification.setdefault("Measurement", []).append(
                "ElectrochemicalImpedanceSpectroscopy" if m == "EIS" else m
            )
            if m in (classification.get("Property") or []):
                try:
                    classification["Property"].remove(m)
                except ValueError:
                    pass
    st.session_state["classification"] = classification

classification = st.session_state.get("classification", {h: [] for h in HEADINGS})
cols = st.columns(3)
for i, h in enumerate(HEADINGS):
    with cols[i % 3]:
        st.markdown(f"**{h}**")
        st.write(", ".join(classification.get(h, [])) or "—")

# ----------------------------
# Template selection (as-is)
# ----------------------------
st.markdown("### Step 3 - Select templates")
with st.expander("Filter", expanded=False):
    filter_text = st.text_input("Filter (search within labels/IDs)", value=st.session_state.get("filter_text", ""))
    st.session_state["filter_text"] = filter_text

ranked: List[TemplateSpec] = rank_templates(TEMPLATES, question, features, classification)

# Apply filter on ranked list
if filter_text.strip():
    ft = filter_text.strip().lower()
    ranked = [t for t in ranked if ft in t.label.lower() or ft in t.id.lower()]

# Build options and a stable map from ID -> label
max_show = 200
ranked_view = ranked[:max_show]
opts_map_by_id: Dict[str, str] = {}
for t in ranked_view:
    score_val = getattr(t, "score", 0.0)
    opts_map_by_id[t.id] = f"{t.label} [{t.id}]  -  score {score_val:.2f}"
opts = list(opts_map_by_id.values())

# Recover previous selected IDs if any; otherwise choose top-N IDs
selected_ids: List[str] = st.session_state.get("tpl_select_ids", [])
if not selected_ids:
    selected_ids = [t.id for t in ranked_view[:10]]
    st.session_state["tpl_select_ids"] = selected_ids

# Compute default labels from IDs, filtering to those present in current options
default_labels: List[str] = [opts_map_by_id[i] for i in selected_ids if i in opts_map_by_id]

# If nothing matches (e.g., strong filter), keep empty default to avoid Streamlit exception
selected_labels = st.multiselect(
    "Suggested templates (hybrid-ranked)",
    options=opts,
    default=default_labels,
    key="tpl_select_labels"
)

# Parse selected IDs from labels and persist them
selected_id_set = {s.split("[")[-1].split("]")[0].strip() for s in (selected_labels or [])}
st.session_state["tpl_select_ids"] = [i for i in selected_ids if i in selected_id_set] + \
                                     [i for i in (ranked_view and [t.id for t in ranked_view]) if i in selected_id_set and i not in selected_ids]

# Build selected TemplateSpec list in current ranking order
selected = [t for t in ranked_view if t.id in st.session_state["tpl_select_ids"]]

with st.expander("Why these?", expanded=False):
    if not ranked:
        st.write("No templates ranked.")
    else:
        for t in ranked[:20]:
            why = f" - {getattr(t,'why','')}" if getattr(t,'why', '') else ""
            s = getattr(t, "score", 0.0)
            st.markdown(f"- **{t.label}** `{t.id}`  (score {s:.2f}){why}")

c1, c2 = st.columns(2)
with c1:
    if st.button("Reset plan (clear selections, triples, values)"):
        for k in ["classification","triples","curated_values","datapoints","template_shapes",
                  "tpl_select_labels","tpl_select_ids"]:
            st.session_state.pop(k, None)
        do_rerun()
with c2:
    if st.button("Apply top 8 suggestions"):
        st.session_state["tpl_select_ids"] = [t.id for t in ranked_view[:8]]
        st.session_state["tpl_select_labels"] = [opts_map_by_id[i] for i in st.session_state["tpl_select_ids"] if i in opts_map_by_id]
        do_rerun()

if st.button("Apply Selected Templates"):
    triples: List[Tuple[str,str,str]] = st.session_state.get("triples", [])
    curated_values: Dict[str, dict] = st.session_state.get("curated_values", {})
    datapoints: List[dict] = st.session_state.get("datapoints", [])
    unit_aliases = st.session_state.get("unit_aliases", load_unit_aliases())
    shapes_to_merge: List[str] = st.session_state.get("template_shapes", [])
    notes_agg: List[str] = []

    for t in selected:
        classification, triples, curated_values, datapoints, notes = apply_template(
            t, classification, triples, curated_values, datapoints
        )
        notes_agg.extend(notes)
        for fp in t.shapes_files:
            shapes_to_merge.append(fp)
        if t.units_map:
            unit_aliases.update(t.units_map)

    st.session_state["classification"] = classification
    st.session_state["triples"] = triples
    st.session_state["curated_values"] = curated_values
    st.session_state["datapoints"] = datapoints
    st.session_state["template_shapes"] = shapes_to_merge
    st.session_state["unit_aliases"] = unit_aliases

    with st.expander("Template notes"):
        if notes_agg:
            for n in notes_agg:
                st.write("• " + n)
        else:
            st.write("Templates applied.")

# ----------------------------
# Relationships editor (as-is)
# ----------------------------
st.markdown("### Step 4 - Relationships (optional)")
triples = st.session_state.get("triples", [])
triples_text = "\n".join(f"{s}  {p}  {o}" for s,p,o in triples)
triples_text = st.text_area("Subject  predicate  Object (two spaces around predicate)", value=triples_text, height=160)
final_triples: List[Tuple[str,str,str]] = []
for line in (triples_text or "").splitlines():
    line = clean_text(line)
    if not line:
        continue
    parts = re.split(r"\s{2,}", line)
    if len(parts) != 3:
        parts = line.split()
        if len(parts) != 3:
            continue
    s,p,o = [x.strip() for x in parts]
    if p in ALLOWED_PREDICATES:
        final_triples.append((s,p,o))
st.session_state["triples"] = final_triples

# ----------------------------
# Value & unit overrides (as-is)
# ----------------------------
st.markdown("### Step 5 - Value & Unit overrides (optional)")
curated_values = st.session_state.get("curated_values", {})
candidates_vu = (classification.get("Property", []) or []) + (classification.get("Parameter", []) or [])
if candidates_vu:
    import pandas as pd
    rows=[]
    for it in sorted(set(candidates_vu)):
        spec = curated_values.get(it, {})
        rows.append({
            "Item": it,
            "numericValue": spec.get("numericValue", None),
            "unitHuman": spec.get("unitHuman", ""),
            "quantityKind": spec.get("quantityKind", ""),
            "normalizationBasis": spec.get("normalizationBasis",""),
            "referenceScale": spec.get("referenceScale",""),
        })
    df = pd.DataFrame(rows)
    edited = st.data_editor(df, use_container_width=True, key="vu_editor")
    curated_values = {}
    for _, r in edited.iterrows():
        curated_values[r["Item"]] = {
            "numericValue": (float(r["numericValue"]) if str(r["numericValue"]).strip() not in ["","None","nan"] else None),
            "unitHuman": (r["unitHuman"] or "").strip(),
            "quantityKind": (r["quantityKind"] or "").strip(),
            "normalizationBasis": (r["normalizationBasis"] or "").strip(),
            "referenceScale": (r["referenceScale"] or "").strip()
        }
    st.session_state["curated_values"] = curated_values

# ----------------------------
# FAIR metadata (as-is)
# ----------------------------
st.markdown("### Step 6 - Dataset FAIR metadata")
with st.form("dataset_meta_form"):
    ds = st.session_state.get("dataset_meta", {})
    ds_title = st.text_input("Dataset title", value=ds.get("title","Hydrogen Experiment Dataset"))
    ds_format= st.text_input("Format", value=ds.get("format","CSV"))
    ds_license = st.text_input("License URL", value=ds.get("license","https://creativecommons.org/licenses/by/4.0/"))
    ds_url   = st.text_input("Download URL", value=ds.get("url","https://example.org/download/dataset.csv"))
    ds_creators = st.text_input("Creators (comma-separated)", value=ds.get("creators","Electrochemistry Research Group"))
    submitted = st.form_submit_button("Save")
if submitted:
    st.session_state["dataset_meta"] = {
        "title": ds_title, "format": ds_format, "license": ds_license,
        "url": ds_url, "creators": ds_creators
    }

# ============================================================================
# A9R — Auto-repair helpers (deterministic, no LLM)
# ============================================================================
_MEAS_HINTS = [
    ("ElectrochemicalImpedanceSpectroscopy", ["EIS","Impedance","ElectrochemicalImpedanceSpectroscopy"]),
    ("LinearSweepVoltammetry", ["LSV","LinearSweep","Linear Sweep","Tafel","Polarization"]),
    ("PolarizationCurve", ["PolarizationCurve","Polarization","IV","I-V"]),
    ("CyclicVoltammetry", ["CV","Cyclic","CyclicVoltammetry"]),
    ("RotatingDiskElectrode", ["RDE","RotatingDisk"]),
    ("RotatingRingDiskElectrode", ["RRDE","Ring-Disk","Ring Disk"]),
    ("Chronoamperometry", ["CA","Chronoamperometry"]),
    ("Chronopotentiometry", ["CP","Chronopotentiometry"])
]

def _infer_meas_kind(name: str) -> str:
    n = (name or "").lower()
    for canon, hints in _MEAS_HINTS:
        for h in hints:
            if h.lower() in n:
                return canon
    for canon, _ in _MEAS_HINTS:
        if canon.lower() == n:
            return canon
    return ""

def _ensure_list(d: Dict[str, List[str]], key: str) -> List[str]:
    d.setdefault(key, [])
    return d[key]

def _add_unique(lst: List[str], item: str) -> None:
    if item not in lst:
        lst.append(item)

def _add_triple_unique(triples: List[Tuple[str,str,str]], s: str, p: str, o: str) -> bool:
    t = (s,p,o)
    if t not in triples:
        triples.append(t); return True
    return False

def _a9r_defaults_for_meas(kind: str) -> Tuple[str, str]:
    k = (kind or "").lower()
    if k in ["electrochemicalimpedancespectroscopy"]:
        return ("ChargeTransferResistance", "FrequencyRange")
    if k in ["linearsweepvoltammetry","polarizationcurve","rotatingdiskelectrode","rotatingringdiskelectrode"]:
        return ("CurrentDensity", "PotentialWindow")
    if k in ["cyclicvoltammetry"]:
        return ("ElectrochemicallyActiveSurfaceArea", "ScanRate")
    if k in ["chronoamperometry"]:
        return ("CurrentDensity", "Duration")
    if k in ["chronopotentiometry"]:
        return ("Overpotential", "Duration")
    return ("CurrentDensity", "ScanRate")

def a9r_auto_repair_once(
    classification: Dict[str, List[str]],
    triples: List[Tuple[str,str,str]],
    curated_values: Dict[str, dict],
    dataset_meta: Dict[str, str]
) -> Tuple[Dict[str, List[str]], List[Tuple[str,str,str]], Dict[str, dict], Dict[str,str], List[str]]:
    notes: List[str] = []

    for k in HEADINGS:
        classification.setdefault(k, [])
        if not isinstance(classification[k], list):
            classification[k] = []

    for i, (s,p,o) in enumerate(list(triples)):
        if p == "usesInstrument" and o not in classification["Instrument"]:
            _add_unique(classification["Instrument"], o)
            notes.append(f"Retyped '{o}' to Instrument (by usesInstrument).")
        if p == "hasParameter" and o not in classification["Parameter"]:
            _add_unique(classification["Parameter"], o)
            notes.append(f"Retyped '{o}' to Parameter (by hasParameter).")
        if p == "measures":
            if o not in classification["Property"]:
                if o in classification["Parameter"]:
                    classification["Parameter"].remove(o)
                _add_unique(classification["Property"], o)
                notes.append(f"Retyped '{o}' to Property (object of measures).")
            if s not in classification["Measurement"]:
                _add_unique(classification["Measurement"], s)
                notes.append(f"Retyped '{s}' to Measurement (subject of measures).")

    if not classification["Instrument"]:
        _add_unique(classification["Instrument"], "ElectrochemicalWorkstation")
        notes.append("Added generic Instrument: ElectrochemicalWorkstation.")

    if classification["Measurement"]:
        for meas in list(classification["Measurement"]):
            kind = _infer_meas_kind(meas) or meas
            default_prop, default_param = _a9r_defaults_for_meas(kind)

            if not any(s == meas and p == "usesInstrument" for s,p,o in triples):
                inst = classification["Instrument"][0]
                if _add_triple_unique(triples, meas, "usesInstrument", inst):
                    notes.append(f"{meas}: added usesInstrument {inst}.")

            has_param_already = any(s == meas and p == "hasParameter" for s,p,o in triples)
            if not has_param_already:
                par = classification["Parameter"][0] if classification["Parameter"] else default_param
                _add_unique(classification["Parameter"], par)
                if _add_triple_unique(triples, meas, "hasParameter", par):
                    notes.append(f"{meas}: added hasParameter {par}.")

            has_measures_already = any(s == meas and p == "measures" for s,p,o in triples)
            if not has_measures_already:
                prop = default_prop
                if classification["Property"] and default_prop not in classification["Property"]:
                    prop = classification["Property"][0]
                _add_unique(classification["Property"], prop)
                if _add_triple_unique(triples, meas, "measures", prop):
                    notes.append(f"{meas}: added measures {prop}.")

    if classification["Measurement"]:
        if not classification["Data"]:
            _add_unique(classification["Data"], "ExperimentDataset")
            notes.append("Added Data node: ExperimentDataset.")
        first_meas = classification["Measurement"][0]
        for d in classification["Data"]:
            if _add_triple_unique(triples, first_meas, "hasOutputData", d):
                notes.append(f"Linked {first_meas} hasOutputData {d}.")

    dataset_meta = dict(dataset_meta or {})
    dataset_meta.setdefault("title", "Hydrogen Experiment Dataset")
    dataset_meta.setdefault("license", "https://creativecommons.org/licenses/by/4.0/")
    dataset_meta.setdefault("url", "https://example.org/download/dataset.csv")
    dataset_meta.setdefault("format", "CSV")
    dataset_meta.setdefault("creators", "Electrochemistry Research Group")

    safe_defaults = {
        "CurrentDensity":       {"unitHuman":"mA/cm2", "quantityKind":"ElectricCurrentDensity", "normalizationBasis":"GeometricArea"},
        "TafelSlope":           {"unitHuman":"mV", "quantityKind":"ElectricPotential"},
        "Overpotential":        {"unitHuman":"mV", "quantityKind":"ElectricPotential", "referenceScale":"RHE"},
        "ChargeTransferResistance":{"unitHuman":"ohm", "quantityKind":"Resistance"},
        "ElectrochemicallyActiveSurfaceArea":{"unitHuman":"m2/g", "quantityKind":"AreaPerMass", "normalizationBasis":"CatalystMass"},
        "ScanRate":             {"unitHuman":"mV", "quantityKind":"ElectricPotential"},
        "PotentialWindow":      {"unitHuman":"V", "quantityKind":"ElectricPotential"},
        "FrequencyRange":       {"unitHuman":"Hz", "quantityKind":"Frequency"},
        "Duration":             {"unitHuman":"s", "quantityKind":"Time"}
    }
    touched = 0
    for item, spec in list(safe_defaults.items()):
        if item in (classification.get("Property", []) + classification.get("Parameter", [])):
            cur = curated_values.get(item, {})
            if not cur.get("unitHuman") and not cur.get("quantityKind") and not cur.get("numericValue"):
                curated_values[item] = dict(spec)
                touched += 1
    if touched:
        notes.append(f"Initialized defaults for {touched} Property/Parameter items (units/quantityKind).")

    for k in classification:
        classification[k] = sorted(list(dict.fromkeys(classification[k])))
    triples = list(dict.fromkeys(triples))

    return classification, triples, curated_values, dataset_meta, notes

# ============================================================================
# Build & Validate (as-is + A9R loop)
# ============================================================================
st.markdown("### Step 7 - Build RDF, validate, export")
if st.button("Build & Validate"):
    unit_aliases = st.session_state.get("unit_aliases", load_unit_aliases())

    # We'll run a small loop: build -> validate -> if fail and A9R is on, repair -> repeat
    repaired_passes = 0
    last_report_text = ""
    last_report_graph = None
    conforms_final = True
    G = None

    while True:
        # Build (always from current session state)
        G = build_graph(
            classification=st.session_state.get("classification", {}),
            relationships=st.session_state.get("triples", []),
            curated_values=st.session_state.get("curated_values", {}),
            datapoints=st.session_state.get("datapoints", []),
            dataset_meta=st.session_state.get("dataset_meta", {}),
            unit_aliases=unit_aliases
        )

        # Validate (optionally)
        if run_shacl:
            extra_shapes = st.session_state.get("template_shapes", [])
            conforms, rgraph, rtext = run_shacl_validation(G, extra_shape_ttls=extra_shapes)
            conforms_final = bool(conforms)
            last_report_graph = rgraph
            last_report_text = rtext

            if conforms_final:
                if repaired_passes == 0:
                    st.success("SHACL: Conforms")
                else:
                    st.success(f"SHACL: Conforms after {repaired_passes} auto-repair pass(es)")
                break

            # If fails and auto-repair is enabled, attempt repair (up to max passes)
            if auto_repair_on_fail and repaired_passes < int(max_repair_passes):
                st.info("SHACL failed. Attempting A9R auto-repair...")
                cls0 = json.loads(json.dumps(st.session_state.get("classification", {})))
                tri0 = list(st.session_state.get("triples", []))
                cur0 = dict(st.session_state.get("curated_values", {}))
                dsm0 = dict(st.session_state.get("dataset_meta", {}))

                cls1, tri1, cur1, dsm1, notes = a9r_auto_repair_once(cls0, tri0, cur0, dsm0)

                # Persist repaired plan
                st.session_state["classification"] = cls1
                st.session_state["triples"] = tri1
                st.session_state["curated_values"] = cur1
                st.session_state.setdefault("dataset_meta", {})
                st.session_state["dataset_meta"].update(dsm1)

                repaired_passes += 1
                with st.expander(f"A9R notes (pass {repaired_passes})", expanded=False):
                    for n in (notes or []):
                        st.write("• " + n)
                # Loop back to rebuild and re-validate
                continue

            # No more repairs (or disabled): stop loop, show report below
            break

        else:
            st.info("SHACL validation disabled.")
            conforms_final = True
            break

    # Show SHACL report if non-conforming
    if run_shacl and not conforms_final:
        st.error("SHACL: Violations found")
        if last_report_graph is not None:
            rep_ttl = last_report_graph.serialize(format="turtle")
            st.download_button(
                "Download SHACL Report (Turtle)",
                rep_ttl,
                file_name="shacl_report.ttl",
                mime="text/turtle",
                key=f"shacl_ttl_pass{repaired_passes}"
            )
        st.text_area(
            "SHACL Report (text)",
            value=last_report_text,
            height=240,
            key=f"shacl_text_pass{repaired_passes}"
        )

        # Graceful "invalid" exports
        try:
            ttl_data_fail = G.serialize(format="turtle")
            st.download_button(
                "Download ABox (Turtle, may not conform)",
                ttl_data_fail,
                file_name=f"hydrogen_abox_invalid_pass{repaired_passes}.ttl",
                mime="text/turtle",
                key=f"ttl_fail_pass{repaired_passes}"
            )
        except Exception as e:
            st.warning(f"Turtle serialization failed: {e}")

        try:
            jsonld_data_fail = G.serialize(format="json-ld", indent=2)
            st.download_button(
                "Download ABox (JSON-LD, may not conform)",
                jsonld_data_fail,
                file_name=f"hydrogen_abox_invalid_pass{repaired_passes}.jsonld",
                mime="application/ld+json",
                key=f"jsonld_fail_pass{repaired_passes}"
            )
        except Exception as e:
            st.warning(f"JSON-LD serialization failed: {e}")

    # Normal exports (only when allowed)
    can_export = (not run_shacl) or conforms_final or (not block_on_shacl)
    if can_export:
        try:
            ttl_data = G.serialize(format="turtle")
            st.download_button("Download ABox (Turtle)", ttl_data, file_name="hydrogen_abox.ttl", mime="text/turtle")
        except Exception as e:
            st.error(f"Turtle serialization failed: {e}")
        try:
            jsonld_data = G.serialize(format="json-ld", indent=2)
            st.download_button("Download ABox (JSON-LD)", jsonld_data, file_name="hydrogen_abox.jsonld", mime="application/ld+json")
        except Exception as e:
            st.error(f"JSON-LD serialization failed: {e}")
    elif run_shacl and not conforms_final:
        st.warning("Export blocked due to SHACL errors. You can uncheck 'Block export if SHACL fails' to enable normal export.")
