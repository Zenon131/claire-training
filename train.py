import glob
from pathlib import Path
from typing import List, Dict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

def load_data_from_files(glob_pattern: str) -> List[Dict[str, str]]:
    """Load text files and convert to training format"""
    data = []
    for file_path in glob.glob(glob_pattern):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # format for instruction fine-tuning
            # modify this based on your data format
            data.append({
                "text": f"<s>[INST] {content} [/INST]</s>"
            })
    return data

def main():
    # config
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # or your base model
    output_dir = "./claire-3b-datasci"
    token = "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"  # your Hugging Face token
    
    # load local files
    training_data = load_data_from_files("./data/**/*.txt")  # adjust glob pattern
    dataset = Dataset.from_list(training_data)

    # load model and tokenizer (CPU-compatible configuration)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map=None,  # Don't use device_map on CPU
        trust_remote_code=True,
        use_auth_token=token,
        low_cpu_mem_usage=True,  # Helps with memory usage on CPU
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # lora config
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # prepare model for training
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # training args (CPU-compatible configuration)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Smaller batch size for CPU
        gradient_accumulation_steps=8,  # Increased for smaller batch size
        learning_rate=2e-4,
        fp16=False,  # Disable fp16 for CPU training
        logging_steps=10,
        save_strategy="epoch",
    )

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([
                torch.tensor(tokenizer.encode(f["text"])) for f in data
            ])
        }
    )

    # train
    trainer.train()
    
    # save
    trainer.save_model()
    
if __name__ == "__main__":
    main()