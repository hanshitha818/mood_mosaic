import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_squared_error
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BIG5_TRAITS = ["O", "C", "E", "A", "N"]

class EssaysBig5Dataset(Dataset):
    def __init__(self, hf_split):
        # hf_split is a Hugging Face Dataset object
        self.hf_split = hf_split

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        row = self.hf_split[idx]
        text = row["text"]
        # O,C,E,A,N columns are strings like "0" or "1" – convert to float
        labels = np.array(
            [float(row[t]) for t in BIG5_TRAITS],
            dtype="float32"
        )
        return text, torch.tensor(labels, dtype=torch.float32)

class PersonalityRegressor(nn.Module):
    def __init__(self,
                 encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_name)
        emb_dim = self.encoder.get_sentence_embedding_dimension()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(BIG5_TRAITS)),
            nn.Tanh(),  # outputs in [-1, 1]
        )

    def forward(self, texts):
        # texts: list of strings
        # Use frozen SBERT: get embeddings as numpy, then convert to torch tensor
        emb_np = self.encoder.encode(texts, convert_to_tensor=False)
        # emb_np is (batch_size, emb_dim)
        emb = torch.tensor(emb_np, dtype=torch.float32, device=DEVICE)
        preds = self.mlp(emb)
        return preds

def collate_batch(batch):
    texts = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch], dim=0)
    return texts, labels

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for texts, labels in loader:
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        preds = model(texts)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for texts, labels in loader:
            labels = labels.to(DEVICE)
            preds = model(texts)
            loss = criterion(preds, labels)
            total_loss += loss.item() * labels.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    return total_loss / len(loader.dataset), rmse

def main():
    print(f"Using device: {DEVICE}")

    # 1. Load HF dataset
    print("Loading jingjietan/essays-big5...")
    ds = load_dataset("jingjietan/essays-big5")

    train_split = ds["train"]
    val_split = ds["validation"]

    print(f"Train rows: {len(train_split)}  Val rows: {len(val_split)}")

    # 2. Wrap in PyTorch datasets
    train_ds = EssaysBig5Dataset(train_split)
    val_ds = EssaysBig5Dataset(val_split)

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=collate_batch
    )

    # 3. Model, optimizer, loss
    model = PersonalityRegressor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.MSELoss()

    # 4. Training loop
    n_epochs = 5
    best_val_loss = float("inf")
    out_dir = Path("experiments/checkpoints/personality_big5")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_big5.pt"

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_rmse = eval_epoch(model, val_loader, criterion)
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val   loss: {val_loss:.4f}  RMSE: {val_rmse:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_name": "sentence-transformers/all-mpnet-base-v2",
                    "big5_traits": BIG5_TRAITS,
                },
                best_path,
            )
            print(f"New best personality model saved to {best_path} (val loss={val_loss:.4f})")

    print("\nDone.")

if __name__ == "__main__":
    main()
