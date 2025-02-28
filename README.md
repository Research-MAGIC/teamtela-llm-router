# Tela Wizard: Building a Multi-Router LLM for High-Quality and Cost-Effective Responses

## TLDR

We introduce a framework for training state-of-the-art LLM routers, systems that dynamically direct queries to either high-quality closed LLMs or cost-effective open-source LLMs, based on query complexity, optimizing both response quality and cost.

## Background

When developing applications using Large Language Models (LLMs), achieving high-quality responses while maintaining a budget is a key challenge. Closed models like GPT-4o provide superior quality but are costly, especially with a high volume of queries. Conversely, Open Source Software (OSS) models are more economical but may not match the quality, especially for complex or domain-specific queries.

An LLM Router helps balance these aspects by deciding which queries are routed to a closed LLM and which to an OSS LLM based on the query's complexity or domain specificity. Below is a schematic representation of an LLM Router:

### LLM Router

Given a set of user queries, an LLM router enables generating high-quality LLM responses while minimizing the overall cost.

## Approach

In this tutorial, we'll demonstrate how to train a causal-LLM classifier on the LambdaLabs platform as an effective LLM router. We make the following design choices:

### Model Choices

We’ll use GPT-4 as an example of a closed LLM and a variety of OSS LLMs, including Mixtral-8x7B, Nvidia Nemotron, Llama 3.1 405b, Llama 3.1 8b, Llama 3.2:3b, Mistral Nemo, Phi4, and others. Our causal LLM classifier will route between these models.

### Response Quality Rating

We'll quantify the quality of an LLM response on a scale of 1 to 5 stars, with higher scores indicating better quality. For simplicity, we'll assume that GPT-4 and GPT-4o and Gemini models always achieves a 5-star rating, so it serves as a reference for other models.

### Causal LLM Classifier

We'll test various models, including Llama 3.2:3b, Llama 3.2:1b, Phi4-mini-instruct, and Phi3-mini-instruct. Our research shows that these models offer superior routing performance compared to smaller architectures.

More concretely, the objective of the causal LLM classifier is to direct "simple" queries to OSS models, thereby maintaining high overall response quality (e.g., an average score of 4.8/5) while significantly reducing costs (e.g., by 50%).

We show that it's possible to build LLM routers that achieve outstanding performance. Below are results from our best-performing LLM routers, the Causal LLM and a Matrix Factorization (MF) model, evaluated on the MT Bench benchmark, which demonstrate that our routers can achieve higher quality with lower costs (i.e., fewer calls to GPT-4) compared to the random baseline and public LLM routing systems from Unify AI and Martian. For more details on these results and additional ones, refer to our paper.

### Benchmark Results

| Benchmark | Performance        |
| --------- | ------------------ |
| MT Bench  | 70% cost reduction |
| MMLU      | 30% cost reduction |
| GSM8K     | 40% cost reduction |

In the following sections, we discuss the steps that enable anyone to build a strong LLM router.

## Table of Contents

1. **Prepare Labeled Data**: The foundation of a robust LLM router is high-quality labeled data. In this section, we'll guide you through preparing this training data.
2. **Test Router Models**: We demonstrate how to test various causal-LLM classifiers using LambdaLabs' API, transforming them into effective LLM routers.
3. **Offline Evaluation**: Using the public codebase (RouteLLM), we will walk through an offline evaluation on standard benchmarks.

### Time to complete

Approximately 20 hours of training using 10k of rows of example, including time to train on a node with 1 H100 and 10 A100 GPUs.

### Data Labeling

We don't have human labels for scores, so we will use the LLM-as-a-Judge approach. GPT-4o-mini will act as an evaluator, reviewing the query and the OSS model's response to provide a score from 1-5. As shown in the paper, the most robust way to get labels is by providing a reference answer for comparison. Here, GPT-4's own response serves as the reference, and the OSS model's response is evaluated against it.

#### Generate OSS Model Responses

```python
import os
from src.online_inference import generate_oss_responses

dataset_df = generate_oss_responses(
    dataset_df, os.getenv("LAMBDALABS_API_KEY"), response_column="oss_response"
)
```

#### Generate GPT-4o-mini-as-a-judge Scores

```python
from src.online_inference import generate_llm_judge_labels

dataset_df = generate_llm_judge_labels(dataset_df, os.getenv('OPENAI_API_KEY'))
```

## Step 2: Test Router Models

In this section, we will explain how to test various causal LLM classifiers to be effective routers. While our data contains `gpt4_response` and `oss_response`, we will only use the pair (query, oss_score) for training. The goal is for the router to rely solely on the query text to determine which model to route to. Our approach is straightforward: we train a 5-way classifier to predict the `oss_score` from the query. At inference time, we will route to an OSS model if our router predicts a high score (i.e., 4-5) and to GPT-4 otherwise.

### Inference Example

Let's show an example of loading the model and running inference with a single example sampled from our data. Note that you need to get access to the models in order to run these evaluations. Let's first show how a formatted input looks like.

```python
# Store your `meta-llama` access token in /home/ray/default/.env with the name LLAMA2_HF_TOKEN
from dotenv import load_dotenv
load_dotenv("/home/ray/default/.env")

from pprint import pprint

# Sample one row from the DataFrame
sampled_row = train_df.sample(n=1, random_state=42)

# Convert the sampled row to a dictionary without the index
input_example = sampled_row.to_dict(orient='records')[0]

print("Prompt:", input_example['prompt'])
print("Label:", input_example['oss_score'])
print("Messages:")
pprint(input_example['messages'])
```

Let's run inference with this example and examine the model's output.

```python
from src.offline_inference import single_example_inference

result = single_example_inference(input_example)
pprint(result)
```

### Benchmark Evaluation

We will use the AutoModelForCausalLM evaluation framework to measure the performance of our router against a random router on GSM8K. We report the percentage of calls the router needs to send to GPT-4 in order to achieve 20%, 50%, and 80% of GPT-4 performance, along with the area under the curve. See our paper for more details on the evaluation metrics.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch  # Necessário para trabalhar com tensores e especificar o dispositivo

from datasets import load_dataset

full_dataset_df = load_dataset("routellm/gpt4_dataset")
train_df = full_dataset_df["train"].to_pandas()

print(f"Train size: {len(train_df)}")
print(train_df.head())

sampled_row = train_df.sample(n=10, random_state=42)

# Convert the sampled row to a dictionary without the index
input_example = sampled_row.to_dict(orient='records')

model_dir = "./merged_model"

model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

#model.half()

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

for example in input_example:
    user_message = {
        "role": "user",
        "content": (
            f"[Question]\n{example['prompt']}\n\nPrediction:\n"
        )
    }

    prompt = f"{system_message['role'].upper()}:\n{system_message['content']}\n\n" \
             f"{user_message['role'].upper()}:\n{user_message['content']}\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    print("****************************\nPrompt:", example['prompt'],"\n***********************\n\n" )
    print("Answer generated:", generated, "\n\n\n******************\n")
```

## Conclusion

In this tutorial, we have successfully built and evaluated a tested LLM router. We generated synthetic labeled data using the LLM-as-a-judge method to train the model, tested various LLM classifiers using LambdaLabs' GPUS and Tela API, and conducted offline evaluation on a standard benchmark-- demonstrating that our model is effective in out-of-domain generalization.
