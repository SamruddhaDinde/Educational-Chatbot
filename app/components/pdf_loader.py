import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter

from app.common.custom_exception import CustomException
from app.common.logger import get_logger

from app.config.config import DATA_PATH, CHUNK_OVERLAP, CHUNK_SIZE
logger = get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("The Data path does not exist")
        
        loader = DirectoryLoader(DATA_PATH,glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        return documents
    except Exception as e:
        error_message = CustomException("Failed to load pdf files", e)
        logger.error(str(error_message))


def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents were found")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)

        text_chunks = text_splitter.split_documents(documents)

        return text_chunks
    except Exception as e:
        error_message= CustomException("Failed to generate chunks", e)
        logger.error(str(error_message))
