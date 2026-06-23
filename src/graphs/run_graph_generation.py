import json
import logging
from itertools import islice
from pathlib import Path

from corpus.iterator import iter_files
from llm_client import LLMProcessor
from settings import settings

from .checkpointing import clear_checkpoint, load_checkpoint, save_checkpoint
from .chunking import chunk_text
from .extraction import deduplicate_entities, deduplicate_relations, extract_from_chunk
from .graph_generator import generate_ages_graph, generate_beings_graph, generate_realms_graph

logger = logging.getLogger(__name__)


def generate_graphs(llm: str | None = None, force: bool = False, max_texts: int | None = None) -> None:
    prompts_path = Path("config/graphs_prompts.json")
    try:
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load prompts from %s", prompts_path)
        return

    graphs_cfg = settings.graphs
    processor = LLMProcessor(
        model_alias=llm,
        use_json_mode=graphs_cfg.use_json_mode,
    )

    logger.info(f"Starting graph generation (model={processor.model_name}, force={force})...")

    files = iter_files(settings.corpus_dir)
    if max_texts is not None:
        files = islice(files, max_texts)

    for file_info in files:
        text_id = file_info.text_id

        book_out_dir = settings.graphs_dir / text_id
        book_out_dir.mkdir(parents=True, exist_ok=True)

        if (book_out_dir / "beings.json").exists() and not force:
            logger.info(f"--- Skipping: {text_id} (already exists) ---")
            continue

        text = file_info.read_text()

        logger.info(f"--- Processing: {text_id} ---")

        chunks = chunk_text(text, max_chars=graphs_cfg.chunk_size, overlap=graphs_cfg.chunk_overlap)
        logger.info(f"Text split into {len(chunks)} chunks.")

        chunk_prompts = {
            "beings": prompts.get("beings", "Extract characters..."),
            "relations": prompts.get("relations", "Extract relations..."),
            "locations": prompts.get("locations", "Extract locations..."),
            "time": prompts.get("time", "Extract time..."),
        }

        results: dict[str, list] = {"beings": [], "relations": [], "locations": [], "times": []}
        start_chunk = 0

        checkpoint = None if force else load_checkpoint(book_out_dir)
        if checkpoint and checkpoint["next_chunk"] <= len(chunks):
            start_chunk = checkpoint["next_chunk"]
            for key in results:
                results[key] = checkpoint.get(key, [])
            logger.info(f"Resuming from chunk {start_chunk + 1}/{len(chunks)} (checkpoint found).")

        for i in range(start_chunk, len(chunks)):
            logger.info(f"  [Chunk {i + 1}/{len(chunks)}] Extracting entities...")
            chunk_results = extract_from_chunk(processor, chunks[i], chunk_prompts)
            for key in results:
                results[key].extend(chunk_results[key])
            save_checkpoint(book_out_dir, i + 1, results)

        all_characters = deduplicate_entities(results["beings"])
        all_relations = deduplicate_relations(results["relations"])
        all_locations = deduplicate_entities(results["locations"])
        all_times = deduplicate_entities(results["times"])
        logger.info(
            f"Extracted unique items: Beings ({len(all_characters)}), Relations ({len(all_relations)}), Locations ({len(all_locations)}), Times ({len(all_times)})"
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

            generate_beings_graph(all_characters, all_relations, book_out_dir)
            generate_realms_graph(all_locations, book_out_dir)
            generate_ages_graph(all_times, book_out_dir)
            clear_checkpoint(book_out_dir)

        except Exception:
            logger.exception("Error saving files or generating graph for %s", text_id)

    logger.info("Graph generation complete.")
