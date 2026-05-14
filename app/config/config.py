import os
HF_TOKEN = os.environ.get("HF_TOKEN")

HUGGINGFACE_REPO_ID = "microsoft/Phi-4-mini-instruct"
ADAPTER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "train", "output", "oop-phi4-lora")
)
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50