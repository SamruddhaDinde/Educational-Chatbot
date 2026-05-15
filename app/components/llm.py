import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN, ADAPTER_PATH
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

        adapter_config = os.path.join(ADAPTER_PATH, "adapter_config.json")
        if os.path.exists(adapter_config):
            logger.info(f"Loading fine-tuned LoRA adapter from {ADAPTER_PATH}")
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
            model = model.merge_and_unload()
        else:
            logger.warning("No fine-tuned adapter found — using base model")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.15,
            return_full_text=False,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>"),
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("LLM loaded successfully")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load LLM", e)
        logger.error(str(error_message))
        return None
