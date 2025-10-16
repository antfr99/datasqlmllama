import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# ---- Environment fixes for Windows ----
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_VISION"] = "1"
os.environ["TRANSFORMERS_NO_AUDIO"] = "1"
os.environ["PEFT_ENABLE_BNB_CPU"] = "0"
os.environ["PEFT_ENABLE_BNB"] = "0"

# --- CONFIG ---
DATA_FILE = "psycho_dataset.jsonl"
OUTPUT_DIR = "./tinylama_psycho_local"
MODEL_PATH = r".\TinyLlama-1.1B-Chat-v1.0"
BLOCK_SIZE = 64             # ⚡ faster, check if dataset fits
BATCH_SIZE = 1              # CPU friendly
GRAD_ACCUM_STEPS = 4        # ⚖️ stable updates, faster effective batch
EPOCHS = 3                  # ⚡ good balance between speed and quality

# --- LOAD TOKENIZER & MODEL ---
print("🔹 Loading TinyLlama model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cpu"
)
model.config.pad_token_id = model.config.eos_token_id

# --- APPLY LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# --- LOAD DATASET ---
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def tokenize_prompt_completion(example):
    full_text = example["prompt"] + example["completion"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=BLOCK_SIZE,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_prompt_completion, remove_columns=["prompt", "completion"])

# --- DATA COLLATOR ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    save_strategy="steps",
    save_steps=200,                # 💾 fewer checkpoints, faster overall
    logging_steps=20,
    save_total_limit=2,
    fp16=False,                    # CPU only
    report_to="none",
    learning_rate=2e-4,            # ⚖️ stable for LoRA small-data tuning
    warmup_ratio=0.05,
)

# --- TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# --- TRAIN ---
print("🚀 Starting optimized LoRA fine-tuning on CPU...")
trainer.train()

# --- SAVE ---
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("🎬 Fine-tuning complete! Model saved at:", OUTPUT_DIR)
