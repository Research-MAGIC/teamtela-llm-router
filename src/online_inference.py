import time
import json
import pandas as pd
import copy
import concurrent.futures
from typing import Dict, Any, List
import openai

from .utils import prepare_llm_queries, prepare_llm_judge_queries, parse_judge_responses


def get_llm_response(
    base_url: str,
    api_key: str,
    llm: str,
    temperature: float,
    max_tokens: int,
    pidx: int,
    messages: List[Dict[str, str]],
    max_retries=1,
    retry_interval=60,
) -> Dict[int, str]:
    """
    Use OpenAI's API to request completions from a specified LLM and manages request retries upon failures.
    """
    retry_count = 0
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    while retry_count <= max_retries:
        try:
            response = client.chat.completions.create(
                model=llm,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return {pidx: response.choices[0].message.content}
        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return {pidx: ""}


def generate_batch_responses(
    base_url: str,
    api_key: str,
    llm: str,
    queries: Dict[int, Any],
    max_concurrent_queries: int,
    temperature: float,
    max_tokens: int,
    verbose: bool = False,
) -> Dict[int, str]:
    """
    This function manages online batch inference of queries using a specified LLM, tracking progress and handling responses.
    """
    print(f"Starting batch inference on {llm} - {len(queries)} queries...")
    queue = copy.copy(queries)
    responses = {}
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_concurrent_queries
    ) as executor:
        future_to_pidx = {}
        while queue or future_to_pidx:
            while queue and len(future_to_pidx) < max_concurrent_queries:
                pidx, messages = queue.popitem()
                future = executor.submit(
                    get_llm_response,
                    base_url,
                    api_key,
                    llm,
                    temperature,
                    max_tokens,
                    pidx,
                    messages,
                )
                future_to_pidx[future] = pidx

            # Collect completed responses
            completed_futures = concurrent.futures.as_completed(future_to_pidx)
            for future in completed_futures:
                pidx = future_to_pidx.pop(future)
                response_dict = future.result()
                responses.update(response_dict)
                if verbose:
                    print(
                        f"# {llm} queries un-processed: {len(queue)}, in-progress: {len(future_to_pidx)}, ready: {1}"
                    )

            # If no futures are completed, wait a bit before checking again
            if not completed_futures:
                time.sleep(0.5)

    print(f"Done in {time.time() - start_time:.2f}sec.")
    return responses


def generate_llama_3_2_3b_response(
    dataset_df: pd.DataFrame,
    api_key: str,
    api_base: str = "http://provider.4090.akashgpu.com:32111/v1",
    response_column: str = "llama3_2_3b_response",
) -> pd.DataFrame:
    """
    Generate llama3.2:3b-instruct-fp16 responses with Tela text generation endpoint
    """
    # Preprocess endpoint queries
    llm_queries = prepare_llm_queries(dataset_df)

    # Online inference
    responses = generate_batch_responses(
        api_base,
        api_key,
        "llama3.2:3b-instruct-fp16",
        llm_queries,
        max_concurrent_queries=10,
        temperature=0.7,
        max_tokens=512,
        verbose=True,
    )

    # Add responses as a column to the dataset
    dataset_df[response_column] = dataset_df.index.map(responses)
    return dataset_df


def generate_phi4_response(
    dataset_df: pd.DataFrame,
    api_key: str,
    api_base: str = "http://provider.a100.hou3.dd.akash.pub:30274/v1",
    response_column: str = "phi4_response",
) -> pd.DataFrame:
    """
    Generate Phi 4 responses with Tela text generation endpoint
    """
    # Preprocess endpoint queries
    llm_queries = prepare_llm_queries(dataset_df)

    # Online inference
    responses = generate_batch_responses(
        api_base,
        api_key,
        "phi4:14b-fp16",
        llm_queries,
        max_concurrent_queries=10,
        temperature=0.7,
        max_tokens=512,
        verbose=True,
    )

    # Add responses as a column to the dataset
    dataset_df[response_column] = dataset_df.index.map(responses)
    return dataset_df


def generate_mistral_nemo_response(
    dataset_df: pd.DataFrame,
    api_key: str,
    api_base: str = "http://provider.a100.hou3.dd.akash.pub:32248/v1",
    response_column: str = "mistral-nemo_response",
) -> pd.DataFrame:
    """
    Generate mistral-nemo:12b-instruct-2407-fp16 responses with Tela text generation endpoint
    """
    # Preprocess endpoint queries
    llm_queries = prepare_llm_queries(dataset_df)

    # Online inference
    responses = generate_batch_responses(
        api_base,
        api_key,
        "mistral-nemo:12b-instruct-2407-fp16",
        llm_queries,
        max_concurrent_queries=10,
        temperature=0.7,
        max_tokens=512,
        verbose=True,
    )

    # Add responses as a column to the dataset
    dataset_df[response_column] = dataset_df.index.map(responses)
    return dataset_df


def generate_llama3_1_8b_response(
    dataset_df: pd.DataFrame,
    api_key: str,
    api_base: str = "http://provider.a100.hou3.dd.akash.pub:30905/v1",
    response_column: str = "llama3-1_8b_response",
) -> pd.DataFrame:
    """
    Generate llama3.1:8b-instruct-fp16 responses with Tela text generation endpoint
    """
    # Preprocess endpoint queries
    llm_queries = prepare_llm_queries(dataset_df)

    # Online inference
    responses = generate_batch_responses(
        api_base,
        api_key,
        "llama3.1:8b-instruct-fp16",
        llm_queries,
        max_concurrent_queries=10,
        temperature=0.7,
        max_tokens=512,
        verbose=True,
    )

    # Add responses as a column to the dataset
    dataset_df[response_column] = dataset_df.index.map(responses)
    return dataset_df


def generate_qween_2_5_32b_response(
    dataset_df: pd.DataFrame,
    api_key: str,
    api_base: str = "http://provider.a100.hou3.dd.akash.pub:32328/v1",
    response_column: str = "qween_2_5_32b_response",
) -> pd.DataFrame:
    """
    Generate qwen2.5:32b-instruct-fp16 responses with Tela text generation endpoint
    """
    # Preprocess endpoint queries
    llm_queries = prepare_llm_queries(dataset_df)

    # Online inference
    responses = generate_batch_responses(
        api_base,
        api_key,
        "qwen2.5:32b-instruct-fp16",
        llm_queries,
        max_concurrent_queries=5,
        temperature=0.7,
        max_tokens=512,
        verbose=True,
    )

    # Add responses as a column to the dataset
    dataset_df[response_column] = dataset_df.index.map(responses)
    return dataset_df


def generate_qween_2_5_72b_response(
    dataset_df: pd.DataFrame,
    api_key: str,
    api_base: str = "http://provider.a100.hou3.dd.akash.pub:30607/v1",
    response_column: str = "qween_2_5_72b_response",
) -> pd.DataFrame:
    """
    Generate qwen2.5:72b-instruct-q8_0 responses with Tela text generation endpoint
    """
    # Preprocess endpoint queries
    llm_queries = prepare_llm_queries(dataset_df)

    # Online inference
    responses = generate_batch_responses(
        api_base,
        api_key,
        "qwen2.5:72b-instruct-q8_0",
        llm_queries,
        max_concurrent_queries=5,
        temperature=0.7,
        max_tokens=512,
        verbose=True,
    )

    # Add responses as a column to the dataset
    dataset_df[response_column] = dataset_df.index.map(responses)
    return dataset_df


def generate_nemotron_70b_response(
    dataset_df: pd.DataFrame,
    api_key: str,
    api_base: str = "https://api.lambdalabs.com/v1",
    response_column: str = "nemotron_70b_response",
) -> pd.DataFrame:
    """
    Generate nemotron:70b-instruct-q8_0 responses with Tela text generation endpoint
    """
    # Preprocess endpoint queries
    llm_queries = prepare_llm_queries(dataset_df)

    # Online inference
    responses = generate_batch_responses(
        api_base,
        "secret_fanhero-dev_92b3f44100bf4454a431a6d1de22d2c7.wChl16h1pGQLbo4C5iEb2nXRMS8SCntW",
        "llama3.1-nemotron-70b-instruct-fp8",
        llm_queries,
        max_concurrent_queries=25,
        temperature=0.7,
        max_tokens=512,
        verbose=True,
    )

    # Add responses as a column to the dataset
    dataset_df[response_column] = dataset_df.index.map(responses)
    return dataset_df


def generate_llm_judge_labels(
    dataset_df: pd.DataFrame,
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    judge_llm: str = "gpt-4o-mini",
    answer_key: str = "phi4_response",
    reference_key: str = "gpt4_response",
    label_key: str = "phi4_score",
    label_explanation_key: str = "phi4_explanation",
    label_precision_key="phi4_fp16",
    label_number_of_parameters_key="phi4_14b",
    precision="fp16",
    number_of_parameters="14b",
) -> pd.DataFrame:
    """
    Generate LLM-as-a-judge labels with OpenAI's API
    """
    with open("assets/judge_template.json") as f:
        judge_template = json.load(f)

    # Preprocess LLM-judge queries
    judge_queries = prepare_llm_judge_queries(
        dataset_df, judge_template, answer_key, reference_key
    )

    # Generate GPT-4o-mini as a judge labels with OpenAI API
    judge_responses = generate_batch_responses(
        api_base,
        api_key,
        judge_llm,
        judge_queries,
        max_concurrent_queries=10,
        temperature=0,
        max_tokens=256,
        verbose=True,
    )

    # Parse judge responses
    labels, explanations = parse_judge_responses(judge_responses)

    # Add judge score as a label column
    dataset_df[label_key] = dataset_df.index.map(labels)
    dataset_df[label_explanation_key] = dataset_df.index.map(explanations)
    dataset_df[label_precision_key] = precision
    dataset_df[label_number_of_parameters_key] = number_of_parameters

    return dataset_df
