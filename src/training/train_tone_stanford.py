import os
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DATASET_NAME = "Cleanlab/stanford-politeness"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "experiments/checkpoints/tone_stanford"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class PolitenessDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def load_politeness_data():
    # Use the test split, which has a binary "label" column (0 = impolite, 1 = polite).
    ds = load_dataset(
        DATASET_NAME,
        split="test",
        data_files={"test": "test.csv"},
    )
    df = ds.to_pandas()

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return X_train, X_val, y_train, y_val


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading Stanford Politeness data...")
    X_train, X_val, y_train, y_val = load_politeness_data()
    print(f"Train size: {len(X_train)}  Val size: {len(X_val)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = PolitenessDataset(X_train, y_train, tokenizer)
    val_ds = PolitenessDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # 0 = impolite, 1 = polite
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_val_acc = 0.0
    num_epochs = 3

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # --------- Train ---------
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # --------- Validate ---------
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                val_losses.append(loss.item())

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = correct / total if total > 0 else 0.0

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val   loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(OUTPUT_DIR, "best_stanford.bin")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokenizer_name": MODEL_NAME,
                    "val_acc": best_val_acc,
                },
                save_path,
            )
            print(
                f"New best tone model saved to {save_path} (val acc={best_val_acc:.4f})"
            )

    print("Done. Best val accuracy:", best_val_acc)


if __name__ == "__main__":
    train()
