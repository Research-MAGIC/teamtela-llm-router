import os
from src.utils import (
    preprocess_nectar,
    load_and_display_nectar,
    save_intermediate_dataset,
)
from src.online_inference import generate_batch_responses, generate_llm_judge_labels
from src.utils import prepare_llm_queries, inspect_llm_judge_queries


def main():
    # Load and preprocess the dataset
    nectar_df = load_and_display_nectar()
    nectar_gpt4_df = preprocess_nectar(
        nectar_df, model="gpt-4", response_column="gpt4_response"
    )
    dataset_df = nectar_gpt4_df.sample(frac=1, random_state=42)

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = "https://api.lambdalabs.com/v1"
    response_column = "llama_3_1_405b_response"
    answer_key = response_column
    label_key = "llama_3_1_405b_score"
    explanation_key = "llama_3_1_405b_explanation"
    precision_key = "llama_3_1_405b_precision"
    number_of_parameters_key = "llama_3_1_405b_number_of_parameters"
    number_of_parameters = "405b"

    # Preprocess endpoint queries
    llm_queries = prepare_llm_queries(dataset_df)

    # Online inference
    responses = generate_batch_responses(
        api_base,
        "secret_fanhero-dev_92b3f44100bf4454a431a6d1de22d2c7.wChl16h1pGQLbo4C5iEb2nXRMS8SCntW",
        "llama3.1-405b-instruct-fp8",
        llm_queries,
        max_concurrent_queries=25,
        temperature=0.7,
        max_tokens=512,
        verbose=True,
    )
    dataset_df[response_column] = dataset_df.index.map(responses)

    inspect_llm_judge_queries(dataset_df, answer_key=answer_key)

    save_intermediate_dataset(dataset_df, "intermediate_nemotron_70b.json")

    # Generate judge labels
    dataset_df = generate_llm_judge_labels(
        dataset_df,
        api_key,
        answer_key=answer_key,
        label_key=label_key,
        label_explanation_key=explanation_key,
        label_precision_key=precision_key,
        label_number_of_parameters_key=number_of_parameters_key,
        number_of_parameters=number_of_parameters,
    )

    # Save intermediate results
    save_intermediate_dataset(dataset_df, "intermediate_nemotron_70b.json")


if __name__ == "__main__":
    main()
