import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# === Logging Setup ===
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# === Configuration ===
DATA_PATH = Path("../data/processed/processed_sessions.parquet")
MAX_SEQ_LEN = 10
BATCH_SIZE = 128
EMBED_DIM = 64
NUM_EPOCHS = 5
TOP_K = 10
MASK_PROB = 0.15


# === Dataset Class ===
class SequenceDataset(Dataset):
    """
    Prepares session-based sequences for training.
    Returns padded input and target sequences.
    """
    def __init__(self, df: pd.DataFrame):
        self.item_set = sorted(df['itemid'].unique())
        self.item2idx = {item: idx + 1 for idx, item in enumerate(self.item_set)}  # 0 = PAD
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        self.sequences = self._build_sequences(df)

    def _build_sequences(self, df):
        sessions = df.sort_values(['visitorid', 'timestamp'])
        grouped = sessions.groupby('visitorid')['itemid'].apply(list)
        return [seq for seq in grouped if len(seq) >= 2]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = [self.item2idx[i] for i in self.sequences[idx] if i in self.item2idx][-MAX_SEQ_LEN:]
        input_seq = seq[:-1]
        target_seq = seq[1:]
        return torch.LongTensor(input_seq), torch.LongTensor(target_seq)


def collate_fn(batch):
    """Pads sequences in a batch for uniform length."""
    x_seqs, y_seqs = zip(*batch)
    x_padded = pad_sequence(x_seqs, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y_seqs, batch_first=True, padding_value=0)
    return x_padded, y_padded


# === SASRec Model ===
class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (SASRec).
    Uses causal attention and position encoding.
    """
    def __init__(self, num_items: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(MAX_SEQ_LEN, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=4, dim_feedforward=128),
            num_layers=2
        )
        self.output = nn.Linear(embed_dim, num_items + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        x = self.embedding(x) + self.position_embedding(positions)
        x = x.permute(1, 0, 2)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        encoded = self.encoder(x, mask=attn_mask).permute(1, 0, 2)
        return self.output(encoded)


# === BERT4Rec Model ===
class BERT4Rec(nn.Module):
    """
    BERT-style masked sequential recommendation model.
    """
    def __init__(self, num_items: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=4, dim_feedforward=128),
            num_layers=2
        )
        self.output = nn.Linear(embed_dim, num_items + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).permute(1, 0, 2)
        out = self.encoder(emb).permute(1, 0, 2)
        return self.output(out)


# === Model Factory ===
def get_model(model_name: Literal['sasrec', 'bert4rec'], num_items: int) -> nn.Module:
    if model_name == "sasrec":
        return SASRec(num_items, embed_dim=EMBED_DIM)
    if model_name == "bert4rec":
        return BERT4Rec(num_items, embed_dim=EMBED_DIM)
    raise ValueError(f"Unknown model: {model_name}")


# === Training Loop ===
def train_model(model: nn.Module, dataloader: DataLoader, num_items: int, device: torch.device, save_path: Path):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, num_items + 1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                logging.info(f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    logging.info(f"Model checkpoint saved to {save_path}")


# === Evaluation Loop ===
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device, top_k: int = TOP_K):
    model.eval()
    hits = total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.topk(logits, top_k, dim=-1).indices
            mask = y != 0
            hits += (preds[mask].eq(y[mask].unsqueeze(-1)).any(dim=-1)).sum().item()
            total += mask.sum().item()

    acc = hits / total if total > 0 else 0
    logging.info(f"Top-{top_k} Accuracy: {acc:.4f}")


# === Main Entry ===
def main():
    df = pd.read_parquet(DATA_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in ["sasrec", "bert4rec"]:
        logging.info(f"\n--- Benchmarking {model_name.upper()} ---")

        dataset = SequenceDataset(df)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        model = get_model(model_name, num_items=len(dataset.item_set))
        model_path = Path(f"models/{model_name}.pt")

        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info(f"Loaded checkpoint for {model_name} from {model_path}.")
        else:
            train_model(model, dataloader, num_items=len(dataset.item_set), device=device, save_path=model_path)

        evaluate_model(model, dataloader, device=device)


if __name__ == "__main__":
    main()