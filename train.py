import pandas as pd
import random
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Load and clean dataset
df = pd.read_csv("legal_text_classification.csv")
df = df.dropna(subset=["case_text", "case_outcome"])
df = df[["case_text", "case_outcome"]].reset_index(drop=True)

# Split into train and validation
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["case_outcome"], random_state=SEED)

def generate_pairs(df, max_pairs_per_class=100):
    examples = []
    labels = df["case_outcome"].unique()

    for label in labels:
        same_class = df[df["case_outcome"] == label]
        other_class = df[df["case_outcome"] != label]

        same_cases = same_class.sample(min(len(same_class), max_pairs_per_class), random_state=SEED)
        other_cases = other_class.sample(min(len(other_class), max_pairs_per_class), random_state=SEED)

        # Positive pairs (same class)
        for i in range(len(same_cases)-1):
            examples.append(InputExample(
                texts=[same_cases.iloc[i]["case_text"], same_cases.iloc[i+1]["case_text"]],
                label=1.0
            ))

        # Negative pairs (different class)
        for i in range(len(other_cases)):
            examples.append(InputExample(
                texts=[same_cases.iloc[i % len(same_cases)]["case_text"], other_cases.iloc[i]["case_text"]],
                label=0.0
            ))

    return examples

# Generate pairs
train_examples = generate_pairs(train_df, max_pairs_per_class=150)
val_examples = generate_pairs(val_df, max_pairs_per_class=50)

# Load model
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Setup data and loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Validation evaluator
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name="val-eval")

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=6,
    warmup_steps=150,
    evaluation_steps=100,
    output_path="./output_legal_similarity_model",
    show_progress_bar=True
)

print("✅ High-Accuracy Training Complete! Model saved to ./output_legal_similarity_model")
