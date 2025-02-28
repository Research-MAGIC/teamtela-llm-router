train.py:

```python
# train.py
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
        # text é uma string (conteúdo do campo "text")
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
        # Delegue quaisquer outros atributos para o tokenizer
        return getattr(self.tokenizer, name)

def main():
    # (A) Verifique quantas GPUs estão disponíveis.
    device_state = PartialState()
    print(f"GPUs disponíveis: {device_state.num_processes}")

    # (B) Carregue o modelo base diretamente na GPU.
    model_name = "unsloth/Llama-3.1-8B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    # Carregue o tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # (C) Adicione tokens especiais para multi-score (1 a 5).
    special_tokens_dict = {
        "additional_special_tokens": ["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # (D) Carregue o dataset local.
    raw_data = datasets.load_dataset(
        "json",
        data_files={"train": "train_data_sample.jsonl"}
    )
    dataset = raw_data["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # (E) Concatene as mensagens em "text"
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

    # Remova as colunas extras que podem causar conflito
    train_dataset = train_dataset.remove_columns(["messages", "prompt"])
    eval_dataset = eval_dataset.remove_columns(["messages", "prompt"])

    # (F) Configure os argumentos de treinamento.
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
    if last_checkpoint is not None:
        print(f"Checkpoint encontrado: {last_checkpoint}. Retomando o treinamento a partir dele.")
    else:
        print("Nenhum checkpoint válido encontrado. Iniciando treinamento do zero.")

    resume_ckpt = last_checkpoint if last_checkpoint is not None else None

    # (G) Crie uma instância da classe de processamento.
    processing = ProcessingClass(tokenizer=tokenizer, max_seq_length=2048)

    # (H) Configure o SFTTrainer utilizando processing_class.
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=processing
    )

    # (I) Inicie o treinamento.
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # (J) Salve o modelo e o tokenizer finalizados.
    trainer.save_model("./Llama_3_1_8B_router/")
    tokenizer.save_pretrained("./Llama_3_1_8B_router/")

if __name__ == "__main__":
    main()
```

MAX_JOBS=24 nohup deepspeed train.py --deepspeed zero_3_optimizer_parameter_offload.json --num_gpus=1 > train.log 2>&1 &

to convert to pytorch:

python zero_to_fp32.py . ./merged_model

test inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Diretório do modelo convertido (FP32 consolidado e, se necessário, convertido para FP16 em memória)
model_dir = "./merged_model"  # ajuste conforme o seu diretório

# Carrega o modelo e o tokenizer
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Converte o modelo para FP16 (se desejado para inferência)
model.half()

# Construa as mensagens do prompt conforme o exemplo
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

# Concatene as mensagens para formar o prompt
prompt = f"{system_message['role'].upper()}:\n{system_message['content']}\n\n" \
         f"{user_message['role'].upper()}:\n{user_message['content']}\n\n"

# Tokenize o prompt e gere a resposta
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id
)

# Pula os tokens de entrada para obter apenas a resposta gerada
generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print("Resposta gerada:", generated)
```

zero_3_optimizer_parameter_offload.json:

```json
{
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "train_batch_size": "auto",
  "gradient_clipping": "auto"
}
```

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

micro-train.py:

```python
# train.py
import os
import torch
import datasets
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer
from accelerate import PartialState
from dataclasses import dataclass

def prepare_subsampled_dataset(n_sample=1000, full_data_file="full_dataset.jsonl", output_file="train_data_sample.jsonl"):
    """
    Carrega o dataset completo a partir de um arquivo JSONL, realiza uma amostragem
    de n_sample exemplos e salva o resultado em output_file.
    """
    if not os.path.exists(full_data_file):
        raise FileNotFoundError(f"O arquivo {full_data_file} não foi encontrado.")

    df = pd.read_json(full_data_file, lines=True)
    subsampled_df = df.sample(n=n_sample, random_state=42)
    subsampled_df.to_json(output_file, orient="records", lines=True)
    print(f"Subamostra de {n_sample} exemplos salva em {output_file}.")

@dataclass
class ProcessingClass:
    tokenizer: AutoTokenizer
    max_seq_length: int

    def __call__(self, text):
        # 'text' é uma string (conteúdo do campo "text")
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
        # Delegue quaisquer outros atributos para o tokenizer
        return getattr(self.tokenizer, name)

def main():
    # Se o arquivo de amostra não existir, gera-o a partir do dataset completo.
    sample_file = "train_data_sample.jsonl"
    full_data_file = "full_dataset.jsonl"  # ajuste conforme seu arquivo de dataset completo
    if not os.path.exists(sample_file):
        print("Arquivo de amostra não encontrado. Gerando subsample a partir do dataset completo...")
        prepare_subsampled_dataset(n_sample=1000, full_data_file=full_data_file, output_file=sample_file)

    # (A) Verifique quantas GPUs estão disponíveis.
    device_state = PartialState()
    print(f"GPUs disponíveis: {device_state.num_processes}")

    # (B) Carregue o modelo base diretamente na GPU.
    model_name = "unsloth/Llama-3.1-8B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    # Carregue o tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Defina o token de padding, se não estiver definido.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # (C) Adicione tokens especiais para multi-score (1 a 5).
    special_tokens_dict = {
        "additional_special_tokens": ["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # (D) Carregue o dataset local (subamostrado).
    raw_data = datasets.load_dataset(
        "json",
        data_files={"train": sample_file}
    )
    dataset = raw_data["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # (E) Concatene as mensagens em um único campo "text"
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

    # Remova as colunas extras que podem causar conflito
    train_dataset = train_dataset.remove_columns(["messages", "prompt"])
    eval_dataset = eval_dataset.remove_columns(["messages", "prompt"])

    # (F) Configure os argumentos de treinamento.
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
    if last_checkpoint is not None:
        print(f"Checkpoint encontrado: {last_checkpoint}. Retomando o treinamento a partir dele.")
    else:
        print("Nenhum checkpoint válido encontrado. Iniciando treinamento do zero.")
    resume_ckpt = last_checkpoint if last_checkpoint is not None else None

    # (G) Crie uma instância da classe de processamento.
    processing = ProcessingClass(tokenizer=tokenizer, max_seq_length=2048)

    # (H) Configure o SFTTrainer utilizando processing_class.
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=processing
    )

    # (I) Inicie o treinamento.
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # (J) Salve o modelo e o tokenizer finalizados.
    trainer.save_model("./Llama_3_1_8B_router/")
    tokenizer.save_pretrained("./Llama_3_1_8B_router/")

if __name__ == "__main__":
    main()
```
