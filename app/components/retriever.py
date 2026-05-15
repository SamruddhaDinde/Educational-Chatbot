from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """<|system|>
You are an educational assistant for Object-Oriented Programming. Answer using ONLY the context provided below. Be concise — 2 to 3 sentences maximum. Do not add lists, bullet points, or extra formatting. If the context is empty or does not contain information relevant to the question, respond with exactly: "I only have information about OOP concepts. Please ask an OOP-related question."<|end|>
<|user|>
Context:
{context}

Question: {question}<|end|>
<|assistant|>
"""


def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])


def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not present or empty")

        llm = load_llm()

        if llm is None:
            raise CustomException("LLM not loaded")

        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.4, "k": 3},
        )

        def format_docs(docs):
            if not docs:
                return ""
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | set_custom_prompt()
            | llm
            | StrOutputParser()
        )

        logger.info("Successfully created the QA chain")
        return chain

    except Exception as e:
        error_message = CustomException("Failed to make a QA chain", e)
        logger.error(str(error_message))
        return None
