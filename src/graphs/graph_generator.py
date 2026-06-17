import json
import logging
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def create_webpage(nodes_for_js, edges_for_js, output_html_path: Path):
    nodes_json = json.dumps(nodes_for_js, ensure_ascii=False, indent=2)
    edges_json = json.dumps(edges_for_js, ensure_ascii=False, indent=2)

    template_path = Path(__file__).parent / "characters_graph.html"
    html_template = template_path.read_text(encoding="utf-8")
    html_content = html_template.replace("{nodes_json}", nodes_json).replace("{edges_json}", edges_json)

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def generate_and_save_graph(personas_data: list, relations_data: list, output_dir: Path):
    personas = pd.DataFrame(personas_data)
    relations = pd.DataFrame(relations_data)

    if not personas.empty:
        personas.rename(columns=lambda x: str(x).strip().title(), inplace=True)
    if not relations.empty:
        relations.rename(columns=lambda x: str(x).strip().title(), inplace=True)

    G = nx.DiGraph()
    node_metadata = {}

    persona_names = (
        set(personas["Name"].dropna().astype(str).str.strip())
        if not personas.empty and "Name" in personas.columns
        else set()
    )

    if not personas.empty and "Name" in personas.columns:
        for _, row in personas.iterrows():
            name = str(row["Name"]).strip()
            if not name:
                continue

            meta = {}
            for col, val in row.items():
                if isinstance(val, list):
                    meta[col] = ", ".join(map(str, val))
                else:
                    meta[col] = val

            node_metadata[name] = meta

    if not relations.empty and "Subject" in relations.columns and "Object" in relations.columns:
        relations = relations.dropna(subset=["Subject", "Object"])
        relations = relations[(relations["Subject"] != "") & (relations["Object"] != "")]

        for _, row in relations.iterrows():
            subj = str(row["Subject"]).strip()
            obj = str(row["Object"]).strip()
            relation_val = row.get("Relation", row.get("relation", ""))

            if subj not in G:
                G.add_node(subj)
            if obj not in G:
                G.add_node(obj)
            G.add_edge(subj, obj, relation=relation_val)
    else:
        logger.warning("No valid relations. The graph will consist of isolated nodes.")

        for name in persona_names:
            if name not in G:
                G.add_node(name)

    degrees = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)

    min_deg = min(degrees.values()) if degrees else 0
    max_deg = max(degrees.values()) if degrees else 0

    def map_size(d):
        if max_deg == min_deg:
            return 4
        return 2 + (d - min_deg) * (6 / (max_deg - min_deg))

    nodes_for_js = []

    for node_id in G.nodes():
        node_id_str = str(node_id)
        meta = node_metadata.get(node_id_str, {})

        node_data = {
            "id": node_id_str,
            "display_name": node_id_str,
            "Category": "Character" if node_id_str in persona_names else "Group",
            "Degree": degrees.get(node_id_str, 0),
            "BetweennessCentrality": betweenness.get(node_id_str, 0.0),
            "size": map_size(degrees.get(node_id_str, 0)),
        }

        for k, v in meta.items():
            if k not in node_data and v not in [None, "", [], "nan", "NaN"] and str(v).lower() != "nan":
                node_data[k] = v

        nodes_for_js.append(node_data)

    edges_for_js = [
        {"source": str(u), "target": str(v), "relation": d.get("relation", "")} for u, v, d in G.edges(data=True)
    ]

    html_target = output_dir / "characters.html"
    create_webpage(nodes_for_js, edges_for_js, html_target)
    logger.info(f"Character graph saved successfully: {html_target}")
