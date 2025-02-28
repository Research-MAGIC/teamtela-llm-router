import pandas as pd

from src.utils import (
    prepare_ft_messages,
    balance_dataset,
)


train_df = pd.read_json("intermediate_nemotron_70b.json")

scores = [
    "nemotron_70b_score",
]

for score in scores:
    train_df["messages"] = prepare_ft_messages(train_df, score)

for score in scores:
    train_df["routing_label"] = train_df[score].apply(lambda x: 1 if x >= 4 else 0)


# here's what the API data format looks like:
print(train_df["messages"].iloc[0])

balanced_train_df = balance_dataset(train_df, key="routing_label")

print(f"Train size: {len(balanced_train_df)}")

output_file = "train_data_sample.jsonl"
n_sample = 10000
max_samples = min(n_sample, len(balanced_train_df))
subsampled_df = balanced_train_df.sample(n=max_samples, random_state=42)
subsampled_df.to_json(output_file, orient="records", lines=True)
