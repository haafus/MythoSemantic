import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corpus.corpus_iterator import normalize_catalog_id
from json_utils import load_json, save_json
from llm_processing import LLMProcessor
from settings import settings

from .graph_generator import generate_and_save_graph

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.json"


def load_checkpoint(book_out_dir: Path) -> dict | None:
    data = load_json(book_out_dir / CHECKPOINT_FILE)
    if isinstance(data, dict) and isinstance(data.get("next_chunk"), int):
        return data
    return None


def save_checkpoint(book_out_dir: Path, next_chunk: int, results: dict) -> None:
    save_json(book_out_dir / CHECKPOINT_FILE, {"next_chunk": next_chunk, **results})


def clear_checkpoint(book_out_dir: Path) -> None:
    (book_out_dir / CHECKPOINT_FILE).unlink(missing_ok=True)


def extract_from_chunk(llm: LLMProcessor, chunk: str, prompts: dict) -> dict[str, list]:
    """Extract all entity types from one chunk.

    Characters, locations and time are independent and run in parallel;
    relations depend on extracted characters and run afterwards.
    """
    with ThreadPoolExecutor(max_workers=3) as pool:
        chars_future = pool.submit(llm.ask_json, prompts["characters"], chunk)
        locs_future = pool.submit(llm.ask_json, prompts["locations"], chunk)
        times_future = pool.submit(llm.ask_json, prompts["time"], chunk)

        try:
            chars = chars_future.result(timeout=600)
        except Exception:
            logger.exception("Failed to extract characters from chunk")
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
        "characters": chars if isinstance(chars, list) else [],
        "relations": rels if isinstance(rels, list) else [],
        "locations": locs if isinstance(locs, list) else [],
        "times": times if isinstance(times, list) else [],
    }


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            split_pos = max(text.rfind("\n", start, end), text.rfind(".", start, end), text.rfind(" ", start, end))
            if split_pos == -1 or split_pos <= start:
                split_pos = end
            else:
                split_pos += 1
        else:
            split_pos = end

        chunks.append(text[start:split_pos])
        next_start = split_pos - overlap
        start = split_pos if next_start <= start else next_start

    return chunks


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


def run_generate_graphs(llm_model: str | None = None, force: bool = False) -> None:
    prompts_path = settings.project_root / "config" / "graphs_prompts.json"
    try:
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load prompts from %s", prompts_path)
        return

    metadata_path = settings.corpus_metadata_path
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return

    try:
        with open(metadata_path, encoding="utf-8") as f:
            corpus = json.load(f)
    except Exception:
        logger.exception("Failed to read metadata")
        return

    graphs_cfg = settings.graphs
    llm = LLMProcessor(
        model_alias=llm_model,
        use_json_mode=graphs_cfg.use_json_mode,
    )

    logger.info(f"Starting graph generation (model={llm.model_name}, force={force})...")

    for book in corpus:
        book_id = book.get("id", "unknown_book")
        text_id = normalize_catalog_id(book_id)
        txt_path = Path(book.get("path", ""))

        book_out_dir = settings.graphs_dir / text_id
        book_out_dir.mkdir(parents=True, exist_ok=True)

        expected_html_path = book_out_dir / "characters.html"

        if expected_html_path.exists():
            if not force:
                logger.info(f"--- Skipping: {book_id} (already exists) ---")
                continue
            else:
                logger.info(f"--- Overwriting: {book_id} (file exists, but force=True is enabled) ---")

        if not txt_path.exists():
            logger.warning(f"Text file not found: {txt_path}")
            continue

        try:
            with open(txt_path, encoding="utf-8") as f:
                text = f.read()
        except Exception:
            logger.exception("Error reading file %s", txt_path)
            continue

        logger.info(f"--- Processing: {book_id} ---")

        chunks = chunk_text(text, max_chars=graphs_cfg.chunk_size, overlap=graphs_cfg.chunk_overlap)
        logger.info(f"Text split into {len(chunks)} chunks.")

        chunk_prompts = {
            "characters": prompts.get("characters", "Extract characters..."),
            "relations": prompts.get("relations", "Extract relations..."),
            "locations": prompts.get("locations", "Extract locations..."),
            "time": prompts.get("time", "Extract time..."),
        }

        results: dict[str, list] = {"characters": [], "relations": [], "locations": [], "times": []}
        start_chunk = 0

        checkpoint = None if force else load_checkpoint(book_out_dir)
        if checkpoint and checkpoint["next_chunk"] <= len(chunks):
            start_chunk = checkpoint["next_chunk"]
            for key in results:
                results[key] = checkpoint.get(key, [])
            logger.info(f"Resuming from chunk {start_chunk + 1}/{len(chunks)} (checkpoint found).")

        for i in range(start_chunk, len(chunks)):
            logger.info(f"  [Chunk {i + 1}/{len(chunks)}] Extracting entities...")
            chunk_results = extract_from_chunk(llm, chunks[i], chunk_prompts)
            for key in results:
                results[key].extend(chunk_results[key])
            save_checkpoint(book_out_dir, i + 1, results)

        all_characters = deduplicate_entities(results["characters"])
        all_relations = deduplicate_relations(results["relations"])
        all_locations = deduplicate_entities(results["locations"])
        all_times = deduplicate_entities(results["times"])
        logger.info(
            f"Extracted unique items: Characters ({len(all_characters)}), Relations ({len(all_relations)}), Locations ({len(all_locations)}), Times ({len(all_times)})"
        )

        try:
            with open(book_out_dir / "personas.json", "w", encoding="utf-8") as f:
                json.dump(all_characters, f, ensure_ascii=False, indent=2)

            with open(book_out_dir / "relations.json", "w", encoding="utf-8") as f:
                json.dump(all_relations, f, ensure_ascii=False, indent=2)

            with open(book_out_dir / "locations.json", "w", encoding="utf-8") as f:
                json.dump(all_locations, f, ensure_ascii=False, indent=2)

            with open(book_out_dir / "times.json", "w", encoding="utf-8") as f:
                json.dump(all_times, f, ensure_ascii=False, indent=2)

            generate_and_save_graph(all_characters, all_relations, book_out_dir)
            clear_checkpoint(book_out_dir)

        except Exception:
            logger.exception("Error saving files or generating graph for %s", book_id)
