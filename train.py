import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import os

# Load and clean dataset
df = pd.read_csv("legal_text_classification.csv")
df = df.dropna(subset=["case_text", "case_outcome"])
df = df[["case_text", "case_outcome"]]

# Generate positive and negative training pairs
examples = []
labels = df["case_outcome"].unique()

for label in labels:
    same_class = df[df["case_outcome"] == label]
    other_class = df[df["case_outcome"] != label]

    same_cases = same_class.sample(min(10, len(same_class)), random_state=42)
    other_cases = other_class.sample(min(10, len(other_class)), random_state=42)

    # Positive pairs
    for i in range(0, len(same_cases)-1):
        examples.append(InputExample(
            texts=[same_cases.iloc[i]["case_text"], same_cases.iloc[i+1]["case_text"]],
            label=1.0
        ))

    # Negative pairs
    for i in range(len(other_cases)):
        examples.append(InputExample(
            texts=[same_cases.iloc[0]["case_text"], other_cases.iloc[i]["case_text"]],
            label=0.0
        ))

# Load model (offline-safe)
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Training setup
train_dataloader = DataLoader(examples, shuffle=True, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100,
    output_path="./output_legal_similarity_model"
)

print("✅ Training Complete! Model saved to ./output_legal_similarity_model")
