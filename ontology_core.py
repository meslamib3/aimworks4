# ontology_core.py
import os
import re
import json
import datetime
from typing import Dict, List, Tuple, Optional, Any

from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, Literal, URIRef, BNode
from pyshacl import validate as shacl_validate

# ===== Namespaces =====
BASE   = Namespace("https://w3id.org/h2kg/hydrogen-ontology#")
PROV   = Namespace("http://www.w3.org/ns/prov#")
SKOS   = Namespace("http://www.w3.org/2004/02/skos/core#")
QUDT   = Namespace("http://qudt.org/schema/qudt/")
UNIT   = Namespace("http://qudt.org/vocab/unit/")
QK     = Namespace("http://qudt.org/vocab/quantitykind/")
DCT    = Namespace("http://purl.org/dc/terms/")
DCAT   = Namespace("http://www.w3.org/ns/dcat#")

HEADINGS = [
    "Matter","Property","Parameter","Manufacturing","Measurement",
    "Instrument","Agent","Data","Metadata","Identifier","Name","Value","Unit"
]

# Only these predicates are accepted in Step 4 (relationships editor)
ALLOWED_PREDICATES = [
    "hasProperty","influences","measures","hasName","hasIdentifier",
    "hasPart","isPartOf","wasAssociatedWith","hasOutputData","usesInstrument",
    "hasInputMaterial","hasOutputMaterial","hasInputData","hasParameter",
    "hasSubProcess","isSubProcessOf","hasValue","hasUnit",
    # DataPoint helpers
    "ofProperty","fromMeasurement","atCurrentDensity","hasQuantityValue",
    "referenceElectrode","normalizedTo"
]

_CAMEL_RE = re.compile(r'(?<=[a-z])(?=[A-Z])')

def clean_text(s: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", (s or "").strip())

def to_camel_case(name: str) -> str:
    name = clean_text(name)
    parts = re.split(r'[^A-Za-z0-9]+', name)
    parts = [p for p in parts if p]
    if not parts:
        return "Unnamed"
    return "".join(p[:1].upper() + p[1:] for p in parts)

def to_valid_id(name: str) -> str:
    s = to_camel_case(name)
    s = re.sub(r'[^A-Za-z0-9]', '', s)
    if not s:
        s = "Entity"
    if s[0].isdigit():
        s = "V" + s
    return s

# ===== TBox loader =====
def load_tbox_graph() -> Graph:
    g = Graph()
    # core ontology (classes, properties)
    core_path = os.path.join("ont", "core_tbox.ttl")
    if os.path.isfile(core_path):
        g.parse(core_path, format="turtle")
    # controlled vocab (e.g., RHE, GeometricArea)
    vocab_path = os.path.join("ont", "vocab.ttl")
    if os.path.isfile(vocab_path):
        g.parse(vocab_path, format="turtle")
    return g

# ===== SHACL validator =====
def _parse_shape(shapes_graph: Graph, path_or_ttl: str) -> None:
    # Try as file path first; if not a file, treat as embedded TTL string
    if path_or_ttl and os.path.isfile(path_or_ttl):
        shapes_graph.parse(path_or_ttl, format="turtle")
    elif path_or_ttl:
        try:
            shapes_graph.parse(data=path_or_ttl, format="turtle")
        except Exception:
            # ignore bad shapes payloads
            pass

def run_shacl_validation(G: Graph, extra_shape_ttls: List[str]) -> Tuple[bool, Graph, str]:
    shapes_graph = Graph()
    core_shapes = os.path.join("shapes", "core_shapes.ttl")
    if os.path.isfile(core_shapes):
        shapes_graph.parse(core_shapes, format="turtle")
    # optional datapoint shape pack
    dp_shapes = os.path.join("shapes", "datapoint_metrics.shacl.ttl")
    if os.path.isfile(dp_shapes):
        shapes_graph.parse(dp_shapes, format="turtle")
    # optional from templates
    for fp in (extra_shape_ttls or []):
        _parse_shape(shapes_graph, fp)

    conforms, rgraph, rtext = shacl_validate(
        data_graph=G, shacl_graph=shapes_graph, inference="rdfs", advanced=True, debug=False
    )
    return bool(conforms), rgraph, rtext

# ===== Unit alias map =====
def load_unit_aliases() -> Dict[str, str]:
    p = os.path.join("config", "units_aliases.json")
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _normalize_unit_string(s: Optional[str]) -> Optional[str]:
    """Map degree symbols etc. to ASCII tokens so we can handle CP-1252 templates safely."""
    if not s:
        return s
    t = str(s)
    # Replace degree-like symbols with 'deg'
    for ch in ["\u00b0", "\u00ba", "\u02da"]:
        t = t.replace(ch, "deg")
    # Replace Ohm symbol with 'ohm'
    t = t.replace("\u03a9", "ohm").replace("\u03c9", "ohm")
    return t.strip()

def qudt_unit_from_human(unit_aliases: Dict[str,str], s: Optional[str]) -> Optional[URIRef]:
    if not s:
        return None
    norm = _normalize_unit_string(s)
    # exact alias first
    if norm in unit_aliases:
        return URIRef(str(UNIT) + unit_aliases[norm])
    # light fallbacks
    if norm and norm.lower() in ["degc", "celsius"]:
        return URIRef(str(UNIT) + "DEG_C")
    if norm and norm.lower() in ["ohm"]:
        return URIRef(str(UNIT) + "OHM")
    if norm and re.match(r"^[A-Za-z0-9\-]+$", norm):
        # looks like a QUDT token already
        return URIRef(str(UNIT) + norm)
    return None

def normalize_quantity_value(item_name: str, numeric: Optional[float], unit_iri: Optional[URIRef]) -> Tuple[Optional[float], Optional[URIRef]]:
    """Light normalizations (e.g., degC -> K for temperature properties)."""
    if unit_iri and str(unit_iri).endswith("/DEG_C"):
        if "temperature" in (item_name or "").lower():
            # convert degC to K if a temperature item
            return (numeric + 273.15) if numeric is not None else None, URIRef(str(UNIT) + "K")
    return numeric, unit_iri

# ===== Graph builder (ABox) =====
def build_graph(
    classification: Dict[str, List[str]],
    relationships: List[Tuple[str,str,str]],
    curated_values: Dict[str, dict],
    datapoints: List[dict],
    dataset_meta: Dict[str, str],
    unit_aliases: Dict[str,str]
) -> Graph:
    G = load_tbox_graph()

    # Bind prefixes
    G.bind("base", BASE)
    G.bind("prov", PROV)
    G.bind("skos", SKOS)
    G.bind("qudt", QUDT)
    G.bind("unit", UNIT)
    G.bind("quantitykind", QK)
    G.bind("dcterms", DCT)
    G.bind("dcat", DCAT)

    # Versioning
    ont_uri = URIRef("https://w3id.org/h2kg/hydrogen-ontology")
    G.add((ont_uri, RDF.type, OWL.Ontology))
    G.add((ont_uri, OWL.versionInfo, Literal(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")))

    CLASS_URIS = {
        "Matter": BASE.Matter, "Property": BASE.Property, "Parameter": BASE.Parameter,
        "Manufacturing": BASE.Manufacturing, "Measurement": BASE.Measurement,
        "Instrument": BASE.Instrument, "Agent": BASE.Agent, "Data": BASE.Data,
        "Metadata": BASE.Metadata
    }
    PRED_URIS = {
        "hasProperty": BASE.hasProperty, "influences": BASE.influences, "measures": BASE.measures,
        "usesInstrument": BASE.usesInstrument, "hasPart": BASE.hasPart, "isPartOf": BASE.isPartOf,
        "hasOutputData": BASE.hasOutputData, "hasInputMaterial": BASE.hasInputMaterial,
        "hasOutputMaterial": BASE.hasOutputMaterial, "hasInputData": BASE.hasInputData,
        "hasParameter": BASE.hasParameter, "hasSubProcess": BASE.hasSubProcess,
        "isSubProcessOf": BASE.isSubProcessOf, "wasAssociatedWith": PROV.wasAssociatedWith,
        "hasQuantityValue": BASE.hasQuantityValue, "referenceElectrode": BASE.referenceElectrode,
        "normalizedTo": BASE.normalizedTo, "ofProperty": BASE.ofProperty,
        "fromMeasurement": BASE.fromMeasurement, "atCurrentDensity": BASE.atCurrentDensity
    }

    individuals: Dict[str, URIRef] = {}

    def mk_ind(name: str, cls: URIRef) -> URIRef:
        key = to_valid_id(name)
        uri = BASE[key]
        # avoid adding type twice
        if uri not in individuals.values():
            individuals[name] = uri
            G.add((uri, RDF.type, cls))
            label = re.sub(_CAMEL_RE, " ", to_camel_case(name))
            G.add((uri, SKOS.prefLabel, Literal(label, lang="en")))
        return uri

    def resolve(name: str, default_cls: Optional[URIRef]=None) -> URIRef:
        # try existing
        for k,v in individuals.items():
            if to_valid_id(k) == to_valid_id(name):
                return v
        # infer class from classification
        cls = default_cls
        for head, lst in (classification or {}).items():
            if name in lst and head in CLASS_URIS:
                cls = CLASS_URIS[head]
                break
        return mk_ind(name, cls or BASE.Metadata)

    # A) materialize classified individuals
    for head, items in (classification or {}).items():
        if head not in CLASS_URIS:
            continue
        for item in items or []:
            mk_ind(item, CLASS_URIS[head])

    # B) relationships
    for s,p,o in (relationships or []):
        if p not in PRED_URIS:
            continue
        su = resolve(s)
        ou = resolve(o)
        G.add((su, PRED_URIS[p], ou))
        if p == "hasOutputData":
            G.add((ou, PROV.wasGeneratedBy, su))
        if p in ["hasInputData", "hasInputMaterial"]:
            G.add((su, PROV.used, ou))

    # C) curated values (Parameters & Properties)
    for item, spec in (curated_values or {}).items():
        if not isinstance(spec, dict):
            continue
        subj = resolve(item)
        unit_iri = qudt_unit_from_human(unit_aliases, spec.get("unitHuman"))
        numeric = spec.get("numericValue")
        qk_tok = (spec.get("quantityKind") or "").strip()
        # light normalization (degC -> K for temperature)
        numeric, unit_iri = normalize_quantity_value(item, numeric, unit_iri)

        if unit_iri is not None:
            qv = BNode()
            G.add((qv, RDF.type, QUDT.QuantityValue))
            G.add((qv, QUDT.unit, unit_iri))
            if numeric is not None:
                G.add((qv, QUDT.numericValue, Literal(float(numeric), datatype=XSD.double)))
            if qk_tok:
                G.add((qv, QUDT.quantityKind, URIRef(str(QK) + qk_tok)))
            G.add((subj, BASE.hasQuantityValue, qv))

        # enums
        rs = (spec.get("referenceScale") or "").strip()
        if rs in ["RHE","SHE","AgAgCl"]:
            G.add((subj, BASE.referenceElectrode, BASE[rs]))
        nb = (spec.get("normalizationBasis") or "").strip()
        if nb in ["GeometricArea","CatalystMass","ECSA","Mass"]:
            G.add((subj, BASE.normalizedTo, BASE[nb]))

    # D) DataPoint construction
    for dp in (datapoints or []):
        if not isinstance(dp, dict):
            continue
        dp_node = BNode()
        G.add((dp_node, RDF.type, BASE.DataPoint))
        ofp = dp.get("ofProperty")
        meas = dp.get("fromMeasurement")
        if ofp:
            G.add((dp_node, BASE.ofProperty, resolve(ofp, BASE.Property)))
        if meas:
            G.add((dp_node, BASE.fromMeasurement, resolve(meas, BASE.Measurement)))

        # value (unit required to make it useful; numeric optional)
        val = dp.get("value") or {}
        if isinstance(val, dict):
            v_unit = qudt_unit_from_human(unit_aliases, val.get("unitHuman"))
            if v_unit is not None:
                vqv = BNode()
                G.add((vqv, RDF.type, QUDT.QuantityValue))
                G.add((vqv, QUDT.unit, v_unit))
                if val.get("numericValue") is not None:
                    G.add((vqv, QUDT.numericValue, Literal(float(val["numericValue"]), datatype=XSD.double)))
                if val.get("quantityKind"):
                    G.add((vqv, QUDT.quantityKind, URIRef(str(QK) + val["quantityKind"])))
                G.add((dp_node, BASE.hasQuantityValue, vqv))

        # conditions
        for cond in (dp.get("conditions") or []):
            if not isinstance(cond, dict):
                continue
            ctype = (cond.get("type") or "").strip().lower()
            # current density condition
            if ctype == "currentdensity" or ctype == "current_density":
                cqv = BNode()
                G.add((cqv, RDF.type, QUDT.QuantityValue))
                unit = qudt_unit_from_human(unit_aliases, cond.get("unitHuman"))
                if unit is not None:
                    G.add((cqv, QUDT.unit, unit))
                G.add((cqv, QUDT.quantityKind, URIRef(str(QK) + "ElectricCurrentDensity")))
                if cond.get("numericValue") is not None:
                    G.add((cqv, QUDT.numericValue, Literal(float(cond["numericValue"]), datatype=XSD.double)))
                G.add((dp_node, BASE.atCurrentDensity, cqv))

        # enums on datapoint
        rs_dp = (dp.get("referenceElectrode") or "").strip()
        if rs_dp in ["RHE","SHE","AgAgCl"]:
            G.add((dp_node, BASE.referenceElectrode, BASE[rs_dp]))
        nb_dp = (dp.get("normalizedTo") or "").strip()
        if nb_dp in ["GeometricArea","CatalystMass","ECSA","Mass"]:
            G.add((dp_node, BASE.normalizedTo, BASE[nb_dp]))

    # E) FAIR metadata for Data nodes
    data_nodes = classification.get("Data", []) if classification else []
    for dn in data_nodes or []:
        du = resolve(dn, BASE.Data)
        title = dataset_meta.get("title") or dn
        G.add((du, DCT.title, Literal(title)))
        if dataset_meta.get("license"):
            lic_val = dataset_meta["license"]
            try:
                G.add((du, DCT.license, URIRef(lic_val)))
            except Exception:
                G.add((du, DCT.license, Literal(lic_val)))
        if dataset_meta.get("url"):
            url_val = dataset_meta["url"]
            try:
                G.add((du, DCAT.downloadURL, URIRef(url_val)))
            except Exception:
                G.add((du, DCAT.downloadURL, Literal(url_val)))
        if dataset_meta.get("format"):
            # IMPORTANT: DCT["format"] (not DCT.format) to avoid Namespace.format collision
            G.add((du, DCT["format"], Literal(dataset_meta["format"])))
        if dataset_meta.get("creators"):
            for c in [x.strip() for x in str(dataset_meta["creators"]).split(",") if x.strip()]:
                G.add((du, DCT.creator, Literal(c)))
        G.add((du, DCT.issued, Literal(datetime.date.today().isoformat(), datatype=XSD.date)))

    return G
