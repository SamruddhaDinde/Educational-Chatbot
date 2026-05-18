
import os
import json
import torch
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "app", ".env"))

MODEL_ID    = "microsoft/Phi-4-mini-instruct"
HF_TOKEN    = os.environ.get("HF_TOKEN")
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "output", "oop-phi4-lora")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")

# ── Test questions with reference answers ─────────────────────────────────────
# These are held-out from training data and used for ROUGE scoring.
TEST_CASES = [
    {
        "question": "Explain the concept of encapsulation with an example.",
        "reference": "Encapsulation bundles data and methods in a class and hides internal state from outside. For example, a BankAccount class keeps balance private and exposes only deposit() and withdraw() methods, preventing arbitrary external modification."
    },
    {
        "question": "How does method overriding differ from method overloading?",
        "reference": "Overriding redefines a parent class method in a subclass with the same signature, resolved at runtime. Overloading defines multiple methods with the same name but different parameter lists in the same class, resolved at compile time."
    },
    {
        "question": "Why is composition preferred over inheritance?",
        "reference": "Composition creates loosely coupled designs by assembling behaviour from separate objects rather than inheriting a fixed hierarchy. It avoids the fragile base class problem and allows behaviour to be changed at runtime by swapping composed objects."
    },
    {
        "question": "What is the purpose of an abstract class?",
        "reference": "An abstract class serves as a base template that cannot be instantiated directly. It enforces that all subclasses implement certain methods while allowing shared concrete behaviour to be defined once in the base class."
    },
    {
        "question": "What does the Single Responsibility Principle mean in practice?",
        "reference": "SRP means each class should do exactly one job. A class handling both database queries and email sending violates SRP — splitting it into two classes means a change in email logic cannot accidentally break database logic."
    },
]

SYSTEM_PROMPT = "You are an educational assistant that explains Object-Oriented Programming concepts clearly and concisely."


def load_model_pipeline(adapter_path: str | None = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()   # merge adapter into base for clean inference

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True,
        return_full_text=False,
    )


def generate_answer(pipe, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = pipe(prompt)
    return result[0]["generated_text"].strip()


def rouge_l_score(hypothesis: str, reference: str) -> float:
    """Simple character-level LCS-based ROUGE-L (no extra dependencies)."""
    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()
    if not h_tokens or not r_tokens:
        return 0.0

    # LCS via DP
    m, n = len(r_tokens), len(h_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if r_tokens[i-1] == h_tokens[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]

    precision = lcs / n if n else 0
    recall    = lcs / m if m else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def plot_loss_curve():
    log_path = os.path.join(ADAPTER_DIR, "training_log.json")
    if not os.path.exists(log_path):
        print("training_log.json not found — skipping loss curve")
        return

    with open(log_path) as f:
        log = json.load(f)

    steps  = [e["step"] for e in log]
    losses = [e["loss"] for e in log]

    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, marker="o", markersize=3, linewidth=1.5, color="steelblue")
    plt.xlabel("Training Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("QLoRA Fine-Tuning Loss — Phi-4-mini-instruct on OOP Data")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(out, dpi=150)
    print(f"Loss curve saved → {out}")


def evaluate():
    if not torch.cuda.is_available():
        raise EnvironmentError("Evaluation requires CUDA. Run on the HPC.")

    print("Loading baseline model (no adapter)...")
    baseline_pipe = load_model_pipeline(adapter_path=None)

    print("Loading fine-tuned model (with LoRA adapter)...")
    finetuned_pipe = load_model_pipeline(adapter_path=ADAPTER_DIR)

    report_lines = [
        "=" * 80,
        "EVALUATION REPORT — Baseline vs QLoRA Fine-Tuned (Phi-4-mini-instruct)",
        "=" * 80,
        "",
    ]
    rouge_scores = []

    for i, case in enumerate(TEST_CASES, 1):
        q   = case["question"]
        ref = case["reference"]

        baseline_ans  = generate_answer(baseline_pipe,  q)
        finetuned_ans = generate_answer(finetuned_pipe, q)

        base_rouge = rouge_l_score(baseline_ans,  ref)
        ft_rouge   = rouge_l_score(finetuned_ans, ref)

        rouge_scores.append({
            "question":       q,
            "baseline_rouge": round(base_rouge, 4),
            "finetuned_rouge": round(ft_rouge, 4),
            "delta":          round(ft_rouge - base_rouge, 4),
        })

        report_lines += [
            f"[{i}] {q}",
            "",
            f"  BASELINE  (ROUGE-L {base_rouge:.3f}):",
            f"  {baseline_ans}",
            "",
            f"  FINE-TUNED (ROUGE-L {ft_rouge:.3f}):",
            f"  {finetuned_ans}",
            "",
            f"  REFERENCE:",
            f"  {ref}",
            "-" * 80,
            "",
        ]

    avg_base = sum(s["baseline_rouge"]  for s in rouge_scores) / len(rouge_scores)
    avg_ft   = sum(s["finetuned_rouge"] for s in rouge_scores) / len(rouge_scores)
    report_lines += [
        f"Average ROUGE-L — Baseline: {avg_base:.4f}  |  Fine-tuned: {avg_ft:.4f}  |  Delta: {avg_ft - avg_base:+.4f}",
        "",
    ]

    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved     → {report_path}")

    rouge_path = os.path.join(OUTPUT_DIR, "rouge_scores.json")
    with open(rouge_path, "w") as f:
        json.dump(rouge_scores, f, indent=2)
    print(f"ROUGE scores     → {rouge_path}")

    plot_loss_curve()


if __name__ == "__main__":
    evaluate()
