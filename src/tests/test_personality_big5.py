import torch
import numpy as np
from sentence_transformers import SentenceTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BIG5_TRAITS = ["O", "C", "E", "A", "N"]

class PersonalityRegressor(torch.nn.Module):
    def __init__(self,
                 encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_name)
        emb_dim = self.encoder.get_sentence_embedding_dimension()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, len(BIG5_TRAITS)),
            torch.nn.Tanh(),  # outputs roughly in [-1, 1]
        )

    def forward(self, texts):
        # Frozen SBERT: embeddings as numpy, then torch tensor
        emb_np = self.encoder.encode(texts, convert_to_tensor=False)
        emb = torch.tensor(emb_np, dtype=torch.float32, device=DEVICE)
        preds = self.mlp(emb)
        return preds

def load_model():
    ckpt_path = "experiments/checkpoints/personality_big5/best_big5.pt"
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = PersonalityRegressor(
        encoder_name=ckpt.get("encoder_name", "sentence-transformers/all-mpnet-base-v2"),
        hidden_dim=128,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def pretty_print_scores(text, scores):
    print("\nText:")
    print("  ", text)
    print("Scores (approx, in [-1, 1]):")
    for trait, val in zip(BIG5_TRAITS, scores):
        print(f"  {trait}: {val:.3f}")

def main():
    model = load_model()

    sample_texts = [
        "I love learning new ideas and exploring different research topics. I enjoy reading, reflecting, and trying creative approaches in my projects.",
        "I always plan my work carefully, make detailed schedules, and double check everything before I submit.",
        "I feel energized when I am around people and I like being the one who starts conversations in group projects.",
        "I try to be patient and supportive with my teammates, and I care a lot about keeping a friendly atmosphere.",
        "I often worry about results and deadlines, and sometimes I feel stressed or anxious when things are uncertain.",
    ]

    with torch.no_grad():
        preds = model(sample_texts).cpu().numpy()

    for text, vec in zip(sample_texts, preds):
        pretty_print_scores(text, vec)

if __name__ == "__main__":
    main()
