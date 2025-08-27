# template_lib.py
import os
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from ontology_core import to_camel_case, clean_text, HEADINGS

@dataclass
class TemplateSpec:
    id: str
    label: str
    description: str = ""
    adds: Dict[str, List[str]] = field(default_factory=dict)
    relationships: List[List[str]] = field(default_factory=list)
    curated_values: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    units_map: Dict[str, str] = field(default_factory=dict)
    shapes_files: List[str] = field(default_factory=list)
    datapoints: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    src_path: Optional[str] = None
    why: Optional[str] = None
    score: float = 0.0

def _is_json(fname: str) -> bool:
    return fname.lower().endswith(".json")

def _safe_join_shapes(shp_path: str, tpl_dir: str) -> str:
    if os.path.isabs(shp_path):
        return shp_path
    repo_shapes = os.path.join("shapes", shp_path)
    if os.path.isfile(repo_shapes):
        return repo_shapes
    return os.path.join(tpl_dir, shp_path)

def load_templates_from_dir(root_dir: str) -> List[TemplateSpec]:
    specs: List[TemplateSpec] = []
    if not os.path.isdir(root_dir):
        return specs
    for dirpath, _, files in os.walk(root_dir):
        for fn in files:
            if not _is_json(fn):
                continue
            fpath = os.path.join(dirpath, fn)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                tpl_dir = os.path.dirname(fpath)
                shapes_resolved = []
                for p in (raw.get("shapes_files") or []):
                    shapes_resolved.append(_safe_join_shapes(p, tpl_dir))
                t = TemplateSpec(
                    id=raw["id"],
                    label=raw["label"],
                    description=raw.get("description", ""),
                    adds=raw.get("adds", {}),
                    relationships=raw.get("relationships", []),
                    curated_values=raw.get("curated_values", {}),
                    units_map=raw.get("units_map", {}),
                    shapes_files=shapes_resolved,
                    datapoints=raw.get("datapoints", []),
                    keywords=raw.get("keywords", []),
                )
                t.src_path = fpath
                specs.append(t)
            except Exception as e:
                bad = TemplateSpec(
                    id=f"PARSE_ERROR::{fn}",
                    label=f"Parse error in {fn}",
                    description=str(e),
                )
                bad.src_path = fpath
                specs.append(bad)
    return specs

_WS_SPLIT = re.compile(r"[^a-z0-9\+\.\-/]+")

def _norm(text: str) -> str:
    return (text or "").lower().replace("–", "-").replace("—", "-").replace("-", "-").strip()

def _tokens(text: str) -> List[str]:
    t = _norm(text)
    if not t:
        return []
    t = t.replace("2","2").replace("3","3").replace("4","4")
    parts = _WS_SPLIT.split(t)
    return [p for p in parts if p]

def _camel_to_words(token: str) -> List[str]:
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', token)
    return _tokens(s)

def _derive_template_tokens(t: TemplateSpec) -> List[str]:
    toks: List[str] = []
    toks += _tokens(t.id)
    toks += _tokens(t.label)
    toks += _tokens(t.description)
    for kw in (t.keywords or []):
        toks += _tokens(kw)
    for _, items in (t.adds or {}).items():
        for it in items or []:
            toks += _camel_to_words(it)
            toks += _tokens(it)
    for tri in (t.relationships or []):
        if len(tri) == 3:
            s, _, o = tri
            toks += _camel_to_words(s) + _tokens(s)
            toks += _camel_to_words(o) + _tokens(o)
    if t.src_path:
        toks += _tokens(os.path.basename(t.src_path))
    seen, out = set(), []
    for x in toks:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

_ALIAS_EXPANSIONS: Dict[str, List[str]] = {
    "rrde": ["rrde","ring-disk","ring disk","ring-disk-electrode","peroxide","h2o2",
             "collection","collection-efficiency","collectionefficiency","ncollection","ncol"],
    "rde": ["rde","rotating-disk","rotating disk","levich","koutecky","kl","k-l"],
    "eis": ["eis","impedance","nyquist","bode","rct","cdl","warburg"],
    "cv":  ["cv","cyclic","cyclic-voltammetry","voltammetry"],
    "lsv": ["lsv","linear","linear-sweep","polarization","tafel"],
    "orr": ["orr","oxygen-reduction","o2 reduction","e1/2","ehalf","peroxide"],
    "oer": ["oer","oxygen-evolution"],
    "her": ["her","hydrogen-evolution"],
    "co2rr": ["co2rr","co2-reduction","co2 reduction","co2","co2r","electroreduction"],
    "gc": ["gc","gas-chromatograph","gas chromatography"],
    "gcms": ["gcms","gc-ms"],
    "hplc": ["hplc"],
    "uvvis": ["uv-vis","uvvis","uv","uv/vis","uv-visible"],
    "am1p5g": ["am1.5g","am15g","am 1.5g","solar-simulator"],
    "koh": ["koh"], "h2so4": ["h2so4"], "hclo4": ["hclo4"], "khco3": ["khco3"], "na2so4": ["na2so4"],
    "rhe": ["rhe"], "she": ["she"], "agagcl": ["ag/agcl","agagcl","ag-agcl"],
    "h2o2": ["h2o2","peroxide"], "ehalf": ["e1/2","half-wave","halfwave"]
}

def _expand_with_aliases(tokens: List[str], raw_text: str = "") -> List[str]:
    expanded = set(tokens)
    rt = raw_text.lower()
    if "collection efficiency" in rt:
        expanded.add("collection-efficiency"); expanded.add("collectionefficiency")
        expanded.add("collection"); expanded.add("efficiency"); expanded.add("rrde")
    for tok in list(tokens):
        for key, words in _ALIAS_EXPANSIONS.items():
            if tok == key or tok in words:
                for w in words:
                    expanded.add(w)
                expanded.add(key)
    return list(expanded)

def _make_query_tokens(question: str, features: List[str]) -> List[str]:
    q = _tokens(question)
    for f in (features or []):
        q += _tokens(f)
        q += _camel_to_words(f)
    q = _expand_with_aliases(q, raw_text=question or "")
    q = [w.replace("co2","co2").replace("o2","o2").replace("h2","h2") for w in q]
    out, seen = [], set()
    for w in q:
        if w and w not in seen:
            seen.add(w); out.append(w)
    return out

def _overlap_score(qtoks: List[str], ttoks: List[str]) -> Tuple[float, List[str]]:
    qset, tset = set(qtoks), set(ttoks)
    ev = sorted(list(qset & tset))
    base = float(len(ev))
    alias_boost = 0.0
    for k, words in _ALIAS_EXPANSIONS.items():
        if any(w in ev for w in words+[k]):
            alias_boost += 0.5
    score = (base ** 0.5) + alias_boost
    return score, ev

def rank_templates(
    templates: List[TemplateSpec],
    question: str,
    features: List[str],
    classification: Dict[str, List[str]]
) -> List[TemplateSpec]:
    ranked: List[TemplateSpec] = []
    qtoks = _make_query_tokens(question or "", features or [])
    cls_measurements = set((classification or {}).get("Measurement", []) or [])

    for t in templates:
        ttoks = _derive_template_tokens(t)
        ttoks = _expand_with_aliases(ttoks)
        s_overlap, evidence = _overlap_score(qtoks, ttoks)

        meas_boost = 0.0
        for item in (t.adds or {}).get("Measurement", []) or []:
            if item in cls_measurements:
                meas_boost += 1.0

        kw_bonus = 0.2 if (t.keywords and len(t.keywords) > 0) else 0.0
        total = s_overlap + meas_boost + kw_bonus
        t.score = float(total)
        t.why = ("matched: " + ", ".join(evidence[:12])) if evidence else None
        ranked.append(t)

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked

def apply_template(
    t: TemplateSpec,
    classification: Dict[str, List[str]],
    triples: List[Tuple[str, str, str]],
    curated_values: Dict[str, dict],
    datapoints: List[dict]
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str, str]], Dict[str, dict], List[dict], List[str]]:
    notes: List[str] = []

    for cls, items in (t.adds or {}).items():
        classification.setdefault(cls, [])
        for it in items:
            if it not in classification[cls]:
                classification[cls].append(it)

    for s, p, o in (t.relationships or []):
        tri = (s, p, o)
        if tri not in triples:
            triples.append(tri)

    for k, v in (t.curated_values or {}).items():
        if k not in curated_values:
            curated_values[k] = v
        else:
            for kk, vv in v.items():
                if vv not in [None, ""]:
                    curated_values[k][kk] = vv

    if t.units_map:
        notes.append(f"Template '{t.label}' provided {len(t.units_map)} unit aliases.")

    for dp in (t.datapoints or []):
        if isinstance(dp, dict):
            datapoints.append(dp)

    for h in list(classification.keys()):
        classification[h] = sorted(list(dict.fromkeys(classification[h])))
    triples = list(dict.fromkeys(triples))
    return classification, triples, curated_values, datapoints, notes
