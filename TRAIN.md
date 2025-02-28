# Tela Wizard: Multi-Router LLM Training Guide

This guide provides a comprehensive walkthrough for training a multi-router LLM using the Tela Wizard framework. The process involves preparing data, configuring the training environment, and executing the training script. This setup is designed to optimize both response quality and cost-efficiency by routing queries to appropriate models based on complexity.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Data Preparation](#data-preparation)
4. [Training the Model](#training-the-model)
5. [Converting and Testing the Model](#converting-and-testing-the-model)
6. [Running Inference](#running-inference)

## Introduction

The Tela Wizard framework is designed to train state-of-the-art LLM routers that dynamically direct queries to either high-quality closed LLMs or cost-effective open-source LLMs. This guide will walk you through the entire process, from data preparation to model training and evaluation.

## Setup

Before starting, ensure you have the necessary dependencies installed. You can install them using the following commands:

```bash
pip install ninja wheel
pip install unsloth
pip install trl
pip install datasets
pip install accelerate
pip install torch torchvision torchaudio
pip install huggingface_hub[cli,hf_transfer]
pip install deepspeed
pip install flash_attn
```

Additionally, ensure you have access to the necessary GPUs. This guide assumes the use of an H100 and 10 A100 GPUs on LambdaLabs.

## Data Preparation

The training process begins with preparing the dataset. The dataset should be in JSONL format and contain fields such as "messages" and "prompt". The `micro-train.py` script includes a function to subsample the dataset if needed.

### Subsampling the Dataset

If you don't have a subsampled dataset, you can generate one using the following function:

```python
def prepare_subsampled_dataset(n_sample=1000, full_data_file="full_dataset.jsonl", output_file="train_data_sample.jsonl"):
    """
    Loads the full dataset from a JSONL file, samples n_sample examples, and saves the result to output_file.
    """
    if not os.path.exists(full_data_file):
        raise FileNotFoundError(f"The file {full_data_file} was not found.")

    df = pd.read_json(full_data_file, lines=True)
    subsampled_df = df.sample(n=n_sample, random_state=42)
    subsampled_df.to_json(output_file, orient="records", lines=True)
    print(f"Subsample of {n_sample} examples saved to {output_file}.")
```

### Running the Subsampling

To run the subsampling, execute the following command:

```bash
python micro-train.py
```

This will check for the existence of the subsampled dataset and generate it if necessary.

## Training the Model

The training script (`train.py`) is designed to fine-tune a pre-trained LLM using the prepared dataset. Below is an overview of the training process:

### Training Script (`train.py`)

```python
import os
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer
from accelerate import PartialState
from dataclasses import dataclass

@dataclass
class ProcessingClass:
    tokenizer: AutoTokenizer
    max_seq_length: int

    def __call__(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length"
        )

    def pad(self, *args, **kwargs):
        return self.tokenizer.pad(*args, **kwargs)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def model_max_length(self):
        return self.tokenizer.model_max_length

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

def main():
    device_state = PartialState()
    print(f"Available GPUs: {device_state.num_processes}")

    model_name = "unsloth/Llama-3.1-8B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    special_tokens_dict = {
        "additional_special_tokens": ["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    raw_data = datasets.load_dataset(
        "json",
        data_files={"train": "train_data_sample.jsonl"}
    )
    dataset = raw_data["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    def concat_messages(example):
        msgs = example.get("messages", [])
        conversation = ""
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            conversation += f"{role.upper()}:\n{content}\n\n"
        example["text"] = conversation
        return example

    train_dataset = train_dataset.map(concat_messages)
    eval_dataset = eval_dataset.map(concat_messages)

    train_dataset = train_dataset.remove_columns(["messages", "prompt"])
    eval_dataset = eval_dataset.remove_columns(["messages", "prompt"])

    training_args = TrainingArguments(
        output_dir="./Llama_3_1_8B_router",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        bf16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        ddp_find_unused_parameters=False,
        deepspeed="./zero_3_optimizer_parameter_offload.json",
    )

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    resume_ckpt = last_checkpoint if last_checkpoint is not None else None

    processing = ProcessingClass(tokenizer=tokenizer, max_seq_length=2048)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=processing
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)

    trainer.save_model("./Llama_3_1_8B_router/")
    tokenizer.save_pretrained("./Llama_3_1_8B_router/")

if __name__ == "__main__":
    main()
```

### Running the Training Script

To start the training process, use the following command:

```bash
MAX_JOBS=24 nohup deepspeed train.py --deepspeed zero_3_optimizer_parameter_offload.json --num_gpus=1 > train.log 2>&1 &
```

This command will run the training script using DeepSpeed for optimization. The training logs will be saved to `train.log`.

## Converting and Testing the Model

After training, you may need to convert the model to a different format for inference. Use the following command to convert the model to PyTorch format:

```bash
python zero_to_fp32.py . ./merged_model
```

## Running Inference

To test the trained model, you can use the following script to run inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "./merged_model"
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

model.half()

system_message = {
    "role": "system",
    "content": (
        "[Instruction]\nBased on the question provided below, predict the score an expert evaluator would give "
        "to an AI assistant's response, considering its helpfulness, relevance, adherence to facts, depth, creativity, "
        "and detail. Your prediction should infer the level of proficiency needed to address the question effectively. "
        "Use a scale from 1 to 5, where a higher score indicates a higher anticipated quality of response. Provide your "
        "prediction as: \"[[predicted rating]]\".\n\n"
        "Score criteria:\n"
        "- **4-5**: The AI assistant can produce a very strong answer, showing deep understanding, creativity, detailed "
        "insight, and high relevance.\n"
        "- **3**: The AI assistant can provide an adequate answer with moderate detail, relevance, and factual accuracy.\n"
        "- **1-2**: The AI assistant will struggle to produce a strong answer due to the question's difficulty, vagueness, or "
        "the assistant's limitations.\n"
    )
}

user_message = {
    "role": "user",
    "content": (
        "[Question]\nWhat challenges did FDR face while in office\n\nPrediction:\n"
    )
}

prompt = f"{system_message['role'].upper()}:\n{system_message['content']}\n\n" \
         f"{user_message['role'].upper()}:\n{user_message['content']}\n\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id
)

generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print("Generated response:", generated)
```

## Conclusion

This guide provides a comprehensive overview of training a multi-router LLM using the Tela Wizard framework. By following these steps, you can optimize response quality and cost-efficiency in your LLM applications. For more details, refer to the accompanying code and configuration files.
