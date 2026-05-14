from langchain_huggingface import HuggingFaceEmbeddings

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    try: 
        model = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5")
        return model
    except Exception as e:
        error_message = CustomException("error while loading the embedding model", e)
        logger.error(str(error_message))
        