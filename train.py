import pandas as pd

from src.utils import (
    prepare_ft_messages,
    balance_dataset,
)


train_df = pd.read_csv("processed_dataset.csv")

scores = [
    "llama3_2_3b_score",
    "llama3_1_8b_score",
    "mistral_nemo_score",
    "qween_2_5_32b_score",
    "nemotron_70b_score",
    "qween_2_5_72b_score",
]

for score in scores:
    train_df["messages"] = prepare_ft_messages(train_df, score)

for score in scores:
    train_df["routing_label"] = train_df[score].apply(lambda x: 1 if x >= 4 else 0)


# here's what the API data format looks like:
print(train_df["messages"].iloc[0])

balanced_train_df = balance_dataset(train_df, key="routing_label")

print(f"Train size: {len(balanced_train_df)}")
