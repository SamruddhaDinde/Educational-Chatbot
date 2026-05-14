import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


def load_llm():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            HUGGINGFACE_REPO_ID,
            token=HF_TOKEN,
        )

        if torch.cuda.is_available():
            logger.info("CUDA detected — loading with 4-bit quantization (BitsAndBytes NF4)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                HUGGINGFACE_REPO_ID,
                token=HF_TOKEN,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            logger.warning("No CUDA detected — loading on CPU, inference will be slow")
            model = AutoModelForCausalLM.from_pretrained(
                HUGGINGFACE_REPO_ID,
                token=HF_TOKEN,
                torch_dtype=torch.float32,
                device_map="cpu",
            )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.5,
            do_sample=True,
            repetition_penalty=1.15,
            return_full_text=False,
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("LLM loaded successfully")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load LLM", e)
        logger.error(str(error_message))
        return None
