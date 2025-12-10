from pathlib import Path
import re

from docling.document_converter import DocumentConverter

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ======================================
# PATHS – RELATIVE TO THIS FILE / REPO
# ======================================

# Repo root (assumes this file lives in the repo root)
ROOT_DIR = Path(__file__).resolve().parent

# Input docs and output chunks are subfolders of the repo
INPUT_DIR = ROOT_DIR / "documents"
OUTPUT_DIR = ROOT_DIR / "chunks"

# Local embedding model (BGE-large) inside the repo
MODEL_DIR = ROOT_DIR / "models" / "bge-large-en-v1.5"


# ==========================
# CHUNKING CONFIG
# ==========================

MAX_CHARS_PER_CHUNK = 4000      # max characters in one chunk
OVERLAP_CHARS = 400             # overlap between chunks (characters)

# Supported file types to process
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".ppt", ".doc"}


# ==========================
# DOC CONVERSION + CHUNKING
# ==========================

def convert_to_markdown(input_path: Path, converter: DocumentConverter) -> str:
    """
    Use Docling to convert a single document to Markdown.
    """
    result = converter.convert(str(input_path))
    md = result.document.export_to_markdown()
    return md


def chunk_markdown(
    markdown_text: str,
    max_chars: int,
    overlap_chars: int,
):
    """
    Simple chunker for Docling's markdown output.

    Strategy:
    - Split by markdown headings (#, ##, ###, ...) so each section is a unit.
    - If a section is too big, split it with a sliding window (with overlap).
    - Returns a list of markdown chunks (strings).
    """
    # Split on headings but keep the heading lines (positive lookahead)
    sections = re.split(r"(?=^#{1,6} )", markdown_text, flags=re.MULTILINE)

    chunks = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        if len(sec) <= max_chars:
            chunks.append(sec)
        else:
            # Section is too big → sliding window with overlap
            step = max_chars - overlap_chars
            if step <= 0:
                raise ValueError("overlap_chars must be smaller than max_chars")

            start = 0
            while start < len(sec):
                end = start + max_chars
                chunk = sec[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += step

    return chunks


def process_folder(
    input_dir: Path,
    output_dir: Path,
    max_chars: int,
    overlap_chars: int,
):
    """
    Walk the input_dir, convert each supported file to markdown,
    chunk it, and write chunk markdown files into output_dir.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Docling converter once
    converter = DocumentConverter()

    for file_path in sorted(input_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Skipping unsupported file type: {file_path.name}")
            continue

        print(f"Converting: {file_path.name}")
        try:
            md = convert_to_markdown(file_path, converter)
        except Exception as e:
            print(f"  ERROR converting {file_path.name}: {e}")
            continue

        if not md.strip():
            print(f"  No content extracted from {file_path.name}, skipping.")
            continue

        # Chunk the markdown
        chunks = chunk_markdown(
            markdown_text=md,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        if not chunks:
            print(f"  No chunks produced for {file_path.name}, skipping.")
            continue

        # Write each chunk as its own markdown file
        stem = file_path.stem  # base name without extension
        for i, chunk in enumerate(chunks, start=1):
            chunk_filename = f"{stem}_chunk-{i:03}.md"
            chunk_path = output_dir / chunk_filename
            with chunk_path.open("w", encoding="utf-8") as f:
                f.write(chunk)
            print(f"  Wrote chunk {i} -> {chunk_filename}")


# ==========================
# EMBEDDINGS + PGVECTOR
# ==========================

def build_pgvector_index(chunks_dir: Path):
    """
    Load markdown chunks from chunks_dir, embed them with BGE-large,
    and store them in a PGVector index.
    """

    # 1. Local embedding model (BGE-large)
    embed_model = HuggingFaceEmbedding(
        model_name=str(MODEL_DIR),
        normalize=True,
    )

    # 2. Configure PGVector store (hard-coded, no env)
    vector_store = PGVectorStore.from_params(
        database="dge",
        host="localhost",
        port="5432",
        user="aakashwalavalkar",
        password="aakash1234",
        table_name="rag_chunks",   # table will be auto-created
        embed_dim=1024             # MUST match BGE-large
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Load .md chunks
    documents = SimpleDirectoryReader(str(chunks_dir)).load_data()

    # 4. Build the index = embedding + inserting into PGVector
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print("RAG index created & stored in PostgreSQL 'dge.rag_chunks'")
    return index


# ==========================
# MAIN PIPELINE
# ==========================

if __name__ == "__main__":
    print(f"Repo root: {ROOT_DIR}")
    print("Step 1: converting documents -> chunks ...")
    process_folder(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        max_chars=MAX_CHARS_PER_CHUNK,
        overlap_chars=OVERLAP_CHARS,
    )

    print("\nStep 2: building PGVector index from chunks ...")
    build_pgvector_index(OUTPUT_DIR)

    print("\nDone.")