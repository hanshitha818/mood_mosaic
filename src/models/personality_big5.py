import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

BIG5_TRAITS = ["O", "C", "E", "A", "N"]


class PersonalityRegressorBig5(nn.Module):
    """
    SBERT + small MLP regressor for Big Five scores, matching the
    architecture used in train_personality_big5.py (hidden_dim=128).
    """

    def __init__(self, encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = SentenceTransformer(encoder_name)
        emb_dim = self.encoder.get_sentence_embedding_dimension()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(BIG5_TRAITS)),
        )

    def forward(self, texts):
        """
        texts: list of strings
        returns: tensor of shape (batch, 5) with raw scores (same scale as training).
        """
        with torch.no_grad():
            emb = self.encoder.encode(texts, convert_to_tensor=True)
        preds = self.mlp(emb)
        return preds
