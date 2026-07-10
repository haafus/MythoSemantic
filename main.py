import os

from clean_gutenberg import clean_gutenberg_texts
from UI import start_home_page
from corpus_builder.build_corpus import build_and_save_corpus
from download_archive import download_and_extract_gdrive
from embedding_analyzer import analyze_embeddings
from embeddings_builder import build_embeddings
from embeddings_clustering.run_clustering import build_clusters
from graphs_generator import run_generate_graphs
import subprocess

from ui_server.run_server import create_app, run_server

app = create_app()

if __name__ == "__main__":
    folder1 = 'chroma_db'
    folder2 = 'cache'
    if not os.path.exists(folder1) or not os.path.exists(folder2):
        print(f"Required folders ('{folder1}' or '{folder2}') are missing. Starting download...")
        GOOGLE_DRIVE_FILE_ID = '1VcqrqgKzENrxDqKqvP93JOUhfCPxMRWw'
        download_and_extract_gdrive(GOOGLE_DRIVE_FILE_ID)
    else:
        print("Both folders already exist in the project root. Download skipped.")
    run_server()