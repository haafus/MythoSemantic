from clean_gutenberg import clean_gutenberg_texts
from UI import start_home_page
from corpus_builder.build_corpus import build_and_save_corpus
from embedding_analyzer import analyze_embeddings
from embeddings_builder import build_embeddings
from embeddings_clustering.run_clustering import build_clusters
from graphs_generator import run_generate_graphs
import subprocess

from ui_server.run_server import create_app, run_server

app = create_app()

if __name__ == "__main__":
    run_server()