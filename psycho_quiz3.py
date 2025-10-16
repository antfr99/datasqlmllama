import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIG ---
MODEL_DIR = "./tinylama_psycho_local"
MAX_TOKENS = 40
REPETITION_PENALTY = 1.2
SHOW_CONFIDENCE = True  # True = display confidence

# --- LOAD MODEL & TOKENIZER ---
print("🔹 Loading fine-tuned TinyLlama model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="cpu")
model.eval()

# --- Helper: estimate confidence ---
def estimate_confidence(logits, generated_ids, prompt_length):
    probs = F.softmax(logits[:, :-1, :], dim=-1)
    gen_token_probs = probs[0, prompt_length - 1:, generated_ids[0, 1:]]
    if len(gen_token_probs) == 0:
        return 0.0
    return torch.mean(gen_token_probs).item()

# --- QUIZ LOOP ---
print("🎬 Psycho quiz bot ready! Type 'exit' to quit.\n")
while True:
    question = input("❓ Ask about Psycho: ").strip()
    if question.lower() == "exit":
        print("👋 Goodbye!")
        break
    if not question:
        continue

    # --- Prompt for model ---
    prompt = f"### Question:\n{question}\n\n### Answer (one short sentence only):"
    inputs = tokenizer(prompt, return_tensors="pt")

    # --- Generate answer ---
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,  # deterministic CPU
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )

    output_ids = output.sequences
    logits = torch.stack(output.scores, dim=1) if output.scores else None

    # --- Decode & clean ---
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()
    if "\n" in answer:
        answer = answer.split("\n")[0].strip()
    if "." in answer:
        answer = answer.split(".")[0].strip() + "."

    # --- Fallback ---
    if answer.lower().startswith("question") or answer == "":
        answer = "I’m not sure about that."

    # --- Confidence ---
    if SHOW_CONFIDENCE and logits is not None:
        conf = estimate_confidence(logits, output_ids, inputs.input_ids.shape[1])
        conf = min(max(conf, 0), 1)
        print(f"🎬 Answer: {answer}  (confidence: {conf:.2f})\n")
    else:
        print(f"🎬 Answer: {answer}\n")
