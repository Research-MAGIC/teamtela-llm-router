import pandas as pd


def concatenate_results():
    # Load all intermediate results
    partial_dfs = [
        pd.read_json("intermediate_llama3_2_3b.json", orient="records"),
        pd.read_json("intermediate_llama3_1_8b.json", orient="records"),
        pd.read_json("intermediate_mistral_nemo.json", orient="records"),
        pd.read_json("intermediate_phi4.json", orient="records"),
        pd.read_json("intermediate_qween_2_5_32b.json", orient="records"),
        pd.read_json("intermediate_qween_2_5_72b.json", orient="records"),
        pd.read_json("intermediate_nemotron_70b.json", orient="records"),
    ]

    # Concatenate all partial DataFrames into one
    final_df = pd.concat(partial_dfs, ignore_index=True)
    final_df.reset_index(drop=True, inplace=True)

    # Save the final DataFrame to disk
    final_df.to_json(
        "processed_dataset.json", orient="records", indent=4, force_ascii=False
    )
    print("Final JSON file saved successfully!")


if __name__ == "__main__":
    concatenate_results()
