import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------
# CONFIG
# -----------------------
INPUT_FILE = "ragas_eval_data.json"       # your input JSON (list of objects)
OUTPUT_FILE = "ragas_results.json"        # output JSON to create in PWD

OLLAMA_MODEL = "qwen2.5:1.5b"

# Local BGE-large model path
EMBED_MODEL_PATH = (
    "/Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/models/"
    "bge-large-en-v1.5"
)

# RAGAS run config: fewer workers + higher timeout to avoid TimeoutError with Ollama
run_config = RunConfig(
    timeout=600,      # up to 600s per operation
    max_retries=3,    # don't hammer endlessly
    max_wait=30,      # max wait between retries
    max_workers=1,    # IMPORTANT: no parallel LLM calls to Ollama
)


# -----------------------
# LOAD JSON DATA
# -----------------------
input_path = Path(INPUT_FILE)
if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError("Expected top-level JSON to be a list of objects.")


# -----------------------
# CONFIGURE LLM + EMBEDDINGS
# -----------------------
llm = Ollama(model=OLLAMA_MODEL)

embed_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_PATH,
    # let HF / torch pick device automatically on Apple Silicon
    model_kwargs={},
    encode_kwargs={"normalize_embeddings": True},
)


# -----------------------
# RUN RAGAS EVALUATION PER SAMPLE
# -----------------------
all_results = []

for idx, sample in enumerate(data):
    print(f"Evaluating sample {idx + 1}/{len(data)}")

    # Each sample is evaluated separately
    ds = Dataset.from_list([sample])

    result = evaluate(
        dataset=ds,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embed_model,
        run_config=run_config,
        raise_exceptions=False,  # don't crash on individual failures
    )

    # In recent Ragas versions, use to_pandas(), NOT to_dict()
    df = result.to_pandas()
    row = df.iloc[0].to_dict()

    # Keep track of original index just in case
    row["sample_index"] = idx

    all_results.append(row)


# -----------------------
# SAVE RESULTS TO JSON
# -----------------------
output_path = Path(OUTPUT_FILE)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\nDone! Saved evaluation results to: {output_path.resolve()}")