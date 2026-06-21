import json
import logging
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_df(data: list, columns_title_case: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(data)
    if not df.empty and columns_title_case:
        df.rename(columns=lambda x: str(x).strip().title(), inplace=True)
    return df


def _extract_names(df: pd.DataFrame) -> set[str]:
    if df.empty or "Name" not in df.columns:
        return set()
    return set(df["Name"].dropna().astype(str).str.strip()) - {""}


def _collect_metadata(df: pd.DataFrame) -> dict[str, dict]:
    metadata = {}
    if df.empty or "Name" not in df.columns:
        return metadata
    for _, row in df.iterrows():
        name = str(row["Name"]).strip()
        if not name:
            continue
        meta = {}
        for col, val in row.items():
            if isinstance(val, list):
                meta[col] = ", ".join(map(str, val))
            else:
                meta[col] = val
        metadata[name] = meta
    return metadata


def _graph_to_json(G: nx.Graph, entity_names: set[str], node_metadata: dict[str, dict], category: str) -> dict:
    degrees = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)

    min_deg = min(degrees.values()) if degrees else 0
    max_deg = max(degrees.values()) if degrees else 0

    def map_size(d):
        if max_deg == min_deg:
            return 4
        return 2 + (d - min_deg) * (6 / (max_deg - min_deg))

    nodes = []
    for node_id in G.nodes():
        node_id_str = str(node_id)
        meta = node_metadata.get(node_id_str, {})
        node_data = {
            "id": node_id_str,
            "display_name": node_id_str,
            "Category": category if node_id_str in entity_names else "Other",
            "Degree": degrees.get(node_id_str, 0),
            "BetweennessCentrality": betweenness.get(node_id_str, 0.0),
            "size": map_size(degrees.get(node_id_str, 0)),
        }
        for k, v in meta.items():
            if k not in node_data and v not in [None, "", [], "nan", "NaN"] and str(v).lower() != "nan":
                node_data[k] = v
        nodes.append(node_data)

    edges = [
        {"source": str(u), "target": str(v), "relation": d.get("relation", "")}
        for u, v, d in G.edges(data=True)
    ]

    return {"nodes": nodes, "edges": edges}


def _save_graph(data: dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Graph saved: {output_path}")


def generate_beings_graph(personas_data: list, relations_data: list, output_dir: Path) -> None:
    personas = _normalize_df(personas_data)
    relations = _normalize_df(relations_data)

    G = nx.DiGraph()
    entity_names = _extract_names(personas)
    node_metadata = _collect_metadata(personas)

    if not relations.empty and "Subject" in relations.columns and "Object" in relations.columns:
        relations = relations.dropna(subset=["Subject", "Object"])
        relations = relations[(relations["Subject"] != "") & (relations["Object"] != "")]
        for _, row in relations.iterrows():
            subj = str(row["Subject"]).strip()
            obj = str(row["Object"]).strip()
            relation_val = row.get("Relation", row.get("relation", ""))
            G.add_node(subj)
            G.add_node(obj)
            G.add_edge(subj, obj, relation=relation_val)
    else:
        logger.warning("No valid relations for beings graph.")
        for name in entity_names:
            G.add_node(name)

    data = _graph_to_json(G, entity_names, node_metadata, "Character")
    _save_graph(data, output_dir / "beings.json")


def _parse_adjacent(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [s.strip() for s in value.replace(";", ",").split(",") if s.strip()]
    return []


def generate_realms_graph(locations_data: list, output_dir: Path) -> None:
    locations = _normalize_df(locations_data)
    G = nx.Graph()
    entity_names = _extract_names(locations)
    node_metadata = _collect_metadata(locations)

    for name in entity_names:
        G.add_node(name)

    adj_col = None
    for col in locations.columns:
        if col.lower().replace(" ", "") in ("adjacentto", "adjacent"):
            adj_col = col
            break

    if adj_col and not locations.empty:
        for _, row in locations.iterrows():
            name = str(row.get("Name", "")).strip()
            if not name:
                continue
            for neighbor in _parse_adjacent(row.get(adj_col)):
                G.add_node(neighbor)
                G.add_edge(name, neighbor, relation="adjacent to")

    data = _graph_to_json(G, entity_names, node_metadata, "Location")
    _save_graph(data, output_dir / "realms.json")


def generate_ages_graph(times_data: list, output_dir: Path) -> None:
    times = _normalize_df(times_data)
    G = nx.Graph()
    entity_names = _extract_names(times)
    node_metadata = _collect_metadata(times)

    for name in entity_names:
        G.add_node(name)

    actor_col = None
    for col in times.columns:
        if col.lower().replace(" ", "") == "keyactors":
            actor_col = col
            break

    if actor_col and not times.empty:
        actor_to_epochs: dict[str, list[str]] = {}
        for _, row in times.iterrows():
            epoch = str(row.get("Name", "")).strip()
            if not epoch:
                continue
            actors_raw = row.get(actor_col)
            if isinstance(actors_raw, str):
                actors = [a.strip() for a in actors_raw.replace(";", ",").split(",") if a.strip()]
            elif isinstance(actors_raw, list):
                actors = [str(a).strip() for a in actors_raw if str(a).strip()]
            else:
                continue
            for actor in actors:
                actor_to_epochs.setdefault(actor, []).append(epoch)

        for actor, epochs in actor_to_epochs.items():
            for i in range(len(epochs)):
                for j in range(i + 1, len(epochs)):
                    if not G.has_edge(epochs[i], epochs[j]):
                        G.add_edge(epochs[i], epochs[j], relation=actor)
                    else:
                        existing = G[epochs[i]][epochs[j]].get("relation", "")
                        G[epochs[i]][epochs[j]]["relation"] = f"{existing}, {actor}"

    data = _graph_to_json(G, entity_names, node_metadata, "Epoch")
    _save_graph(data, output_dir / "ages.json")
