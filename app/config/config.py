import os
HF_TOKEN = os.environ.get("HF_TOKEN")
#GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

HUGGINGFACE_REPO_ID="microsoft/Phi-4-mini-instruct"
DB_FAISS_PATH="vectorstore/db_faiss"
DATA_PATH="data/"
CHUNK_SIZE=500
CHUNK_OVERLAP=50