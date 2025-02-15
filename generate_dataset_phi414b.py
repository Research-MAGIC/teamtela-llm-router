import os
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils import (
    preprocess_nectar,
    load_and_display_nectar,
    inspect_llm_judge_queries,
)
from src.online_inference import (
    generate_llm_judge_labels,
    generate_llama_3_2_3b_response,
    generate_llama3_1_8b_response,
    generate_phi4_response,
    generate_mistral_nemo_response,
    generate_qween_2_5_32b_response,
    generate_qween_2_5_72b_response,
    generate_nemotron_70b_response,
    save_intermediate_dataset,
)


def process_model(
    dataset_df,
    api_key,
    generate_func,
    response_column,
    answer_key,
    label_key,
    explanation_key,
    precision_key,
    number_of_parameters_key,
    number_of_parameters,
    precision=None,
    max_retries=3,
    retry_interval=5,
):
    """Process a single model by generating responses and labels with retry logic."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            dataset_df = generate_func(
                dataset_df, api_key, response_column=response_column
            )
            inspect_llm_judge_queries(dataset_df, answer_key=answer_key)
            dataset_df = generate_llm_judge_labels(
                dataset_df,
                api_key,
                answer_key=answer_key,
                label_key=label_key,
                label_explanation_key=explanation_key,
                label_precision_key=precision_key,
                label_number_of_parameters_key=number_of_parameters_key,
                number_of_parameters=number_of_parameters,
                precision=precision,
            )
            return dataset_df
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing {generate_func.__name__}: {e}")
            retry_count += 1
            time.sleep(retry_interval)
    raise Exception(
        f"Failed to process {generate_func.__name__} after {max_retries} retries"
    )


# Load and preprocess the dataset
nectar_df = load_and_display_nectar()
nectar_gpt4_df = preprocess_nectar(
    nectar_df, model="gpt-4", response_column="gpt4_response"
)
dataset_df = nectar_gpt4_df.sample(n=1, random_state=42)
# dataset_df = nectar_gpt4_df.sample(frac=1, random_state=42)

# Define the models and their parameters
models = [
    (
        "generate_llama_3_2_3b_response",
        "llama3_2_3b_response",
        "llama3_2_3b_score",
        "llama3_2_3b_explanation",
        "llama3_2_3b-precision",
        "llama3_2_3b_number_of_parameters",
        "3b",
    ),
    (
        "generate_llama3_1_8b_response",
        "llama3_1_8b_response",
        "llama3_1_8b_score",
        "llama3_1_8b_explanation",
        "llama3_1_8b_precision",
        "llama3_1_8b_number_of_parameters",
        "8b",
    ),
    (
        "generate_mistral_nemo_response",
        "mistral_nemo_response",
        "mistral_nemo_score",
        "mistral_nemo_explanation",
        "mistral_nemo_precision",
        "mistral_nemo_number_of_parameters",
        "12b",
    ),
    ("generate_phi4_response", "phi4_response", None, None, None, None, None),
    (
        "generate_qween_2_5_32b_response",
        "qween_2_5_32b_response",
        "qween_2_5_32b_score",
        "qween_2_5_32b_explanation",
        "qween_2_5_32b_precision",
        "qween_2_5_32b_number_of_parameters",
        "32b",
    ),
    (
        "generate_nemotron_70b_response",
        "nemotron_70b_response",
        "nemotron_70b_score",
        "nemotron_70b_explanation",
        "nemotron_70b_precision",
        "nemotron_70b_number_of_parameters",
        "70b",
        "q8",
    ),
    (
        "generate_qween_2_5_72b_response",
        "qween_2_5_72b_response",
        "qween_2_5_72b_score",
        "qween_2_5_72b_explanation",
        "qween_2_5_72b_precision",
        "qween_2_5_72b_number_of_parameters",
        "72b",
        "q8",
    ),
]

# Process each model in parallel
api_key = os.getenv("OPENAI_API_KEY")
try:
    with ThreadPoolExecutor() as executor:
        futures = []
        for model in models:
            # Ensure the function is available in the global scope
            if model[0] in globals():
                generate_func = globals()[model[0]]
                futures.append(
                    executor.submit(
                        process_model,
                        dataset_df.copy(),
                        api_key,
                        generate_func,
                        *model[1:],
                    )
                )
            else:
                print(f"Function {model[0]} not found in global scope.")

        for future in as_completed(futures):
            result_df = future.result()
            # Save intermediate state for each model
            save_intermediate_dataset(result_df, f"intermediate_{model[1]}.json")

except Exception as e:
    traceback.print_exc()
    print(f"An error occurred: {e}")

# Save the final DataFrame to disk
output_file = "processed_dataset.json"
dataset_df.reset_index(drop=True, inplace=True)
dataset_df.to_json(output_file, orient="records", indent=4, force_ascii=False)
print("Final JSON file saved successfully!")
