# train.py
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
        # text is a string (content of the "text" field)
        return self.tokenizer(
            text, truncation=True, max_length=self.max_seq_length, padding="max_length"
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
        # Delegate any other attributes to the tokenizer
        return getattr(self.tokenizer, name)


def main():
    # (A) Check how many GPUs are available.
    device_state = PartialState()
    print(f"GPUs available: {device_state.num_processes}")

    # (B) Load the base model directly on the GPU.
    model_name = "microsoft/Phi-4-mini-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set a padding token if it is not defined.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # (C) Add special tokens for multi-score (1 to 5).
    special_tokens_dict = {
        "additional_special_tokens": ["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # (D) Load the local dataset.
    raw_data = datasets.load_dataset("json", data_files={"train": "full_dataset.jsonl"})
    dataset = raw_data["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # (E) Concatenate the messages into "text"
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

    # Remove extra columns that might cause conflicts
    train_dataset = train_dataset.remove_columns(["messages", "prompt"])
    eval_dataset = eval_dataset.remove_columns(["messages", "prompt"])

    # (F) Set up the training arguments.
    training_args = TrainingArguments(
        output_dir="./phi4_mini_router",
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
    if last_checkpoint is not None:
        print(f"Checkpoint found: {last_checkpoint}. Resuming training from it.")
    else:
        print("No valid checkpoint found. Starting training from scratch.")

    resume_ckpt = last_checkpoint if last_checkpoint is not None else None

    # (G) Create an instance of the processing class.
    processing = ProcessingClass(tokenizer=tokenizer, max_seq_length=2048)

    # (H) Set up the SFTTrainer using the processing_class.
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=processing,
    )

    # (I) Start training.
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # (J) Save the final model and tokenizer.
    trainer.save_model("./phi4_mini_router/")
    tokenizer.save_pretrained("./phi4_mini_router/")


if __name__ == "__main__":
    main()
