import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from llm_client import LLMProcessor

logger = logging.getLogger(__name__)


def extract_from_chunk(llm: LLMProcessor, chunk: str, prompts: dict) -> dict[str, list]:
    """Extract all entity types from one chunk.

    Characters, locations and time are independent and run in parallel;
    relations depend on extracted characters and run afterwards.
    """
    with ThreadPoolExecutor(max_workers=3) as pool:
        chars_future = pool.submit(llm.ask_json, prompts["beings"], chunk)
        locs_future = pool.submit(llm.ask_json, prompts["locations"], chunk)
        times_future = pool.submit(llm.ask_json, prompts["time"], chunk)

        try:
            chars = chars_future.result(timeout=600)
        except Exception:
            logger.exception("Failed to extract beings from chunk")
            chars = []

        relations_content = (
            f"DOCUMENT 1 (Text):\n{chunk}\n\n"
            f"DOCUMENT 2 (Characters):\n{json.dumps(chars, ensure_ascii=False)}"
        )
        rels = llm.ask_json(prompts["relations"], relations_content)

        try:
            locs = locs_future.result(timeout=600)
        except Exception:
            logger.exception("Failed to extract locations from chunk")
            locs = []

        try:
            times = times_future.result(timeout=600)
        except Exception:
            logger.exception("Failed to extract time from chunk")
            times = []

    return {
        "beings": chars if isinstance(chars, list) else [],
        "relations": rels if isinstance(rels, list) else [],
        "locations": locs if isinstance(locs, list) else [],
        "times": times if isinstance(times, list) else [],
    }


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    unique_entities = {}
    for ent in entities:
        name = ent.get("Name") or ent.get("name") or ent.get("NAME")
        if not name:
            continue

        norm_name = str(name).strip().lower()

        if norm_name not in unique_entities:
            unique_entities[norm_name] = copy.deepcopy(ent)
        else:
            existing = unique_entities[norm_name]
            for key, value in ent.items():
                if key.lower() in ["name"]:
                    continue

                if value in [None, "", [], "nan", "NaN"] or str(value).lower() == "nan":
                    continue

                existing_val = existing.get(key)

                if existing_val in [None, "", [], "nan", "NaN"] or str(existing_val).lower() == "nan":
                    existing[key] = copy.deepcopy(value)
                elif isinstance(existing_val, list) and isinstance(value, list):
                    merged = existing_val.copy()
                    for item in value:
                        if item not in merged:
                            merged.append(item)
                    existing[key] = merged
                elif isinstance(existing_val, str) and isinstance(value, str):
                    if value not in existing_val:
                        existing[key] = f"{existing_val}; {value}"
                elif isinstance(existing_val, list) and isinstance(value, str):
                    if value not in existing_val:
                        existing[key].append(value)
                elif isinstance(existing_val, str) and isinstance(value, list):
                    new_list = [existing_val]
                    for item in value:
                        if item not in new_list:
                            new_list.append(item)
                    existing[key] = new_list

    return list(unique_entities.values())


def deduplicate_relations(relations: list[dict]) -> list[dict]:
    unique_relations = set()
    deduplicated = []
    for rel in relations:
        subj = str(rel.get("Subject", rel.get("subject", ""))).strip().lower()
        obj = str(rel.get("Object", rel.get("object", ""))).strip().lower()
        r_type = str(rel.get("Relation", rel.get("relation", ""))).strip().lower()

        if not subj or not obj:
            continue

        identifier = (subj, obj, r_type)
        if identifier not in unique_relations:
            unique_relations.add(identifier)
            deduplicated.append(rel)

    return deduplicated
