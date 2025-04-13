# api_service.py

"""
FastAPI-based recommendation service for session-based models.
Supports /recommend and /similar endpoints using SQLite mock DB.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import torch
import sqlite3
from pathlib import Path
from typing import List
import logging

import pandas as pd
from torch.utils.data import DataLoader

from models.train_sequence_models import SequenceDataset, get_model, MAX_SEQ_LEN

# === App Setup ===
app = FastAPI(title="Recommendation API")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = Path("models/sasrec.pt")
DATA_PATH = Path("data/processed/processed_sessions.parquet")
DB_PATH = Path("mockshop.db")

# === Load Data & Model ===
df = pd.read_parquet(DATA_PATH)
dataset = SequenceDataset(df)
item_set = dataset.item_set
item2idx = dataset.item2idx
idx2item = dataset.idx2item

model = get_model("sasrec", num_items=len(item_set))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === SQLite Setup ===
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    session TEXT
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    category TEXT,
    title TEXT
);
""")
conn.commit()


# === Helper Functions ===
def recommend_items(user_id: int, k: int = 10) -> List[int]:
    cursor.execute("SELECT session FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    session = eval(row[0])
    input_ids = [item2idx.get(i, 0) for i in session if i in item2idx]
    if not input_ids:
        raise HTTPException(status_code=400, detail="No valid items in session")

    x = torch.LongTensor(input_ids[-MAX_SEQ_LEN:]).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        scores = torch.topk(logits[:, -1, :], k=k).indices.squeeze().tolist()
        return [idx2item[i] for i in scores if i in idx2item]


def similar_items(product_id: int, k: int = 10) -> List[int]:
    if product_id not in item2idx:
        raise HTTPException(status_code=404, detail="Item not found")

    x = torch.LongTensor([item2idx[product_id]]).repeat(1, MAX_SEQ_LEN).to(device)
    with torch.no_grad():
        logits = model(x)
        scores = torch.topk(logits[:, -1, :], k=k).indices.squeeze().tolist()
        return [idx2item[i] for i in scores if i in idx2item and idx2item[i] != product_id]


# === API Endpoints ===
@app.get("/recommend")
def recommend(user_id: int = Query(..., description="User ID"), top_k: int = 10):
    """Recommend items based on user's recent session."""
    return {"user_id": user_id, "recommendations": recommend_items(user_id, top_k)}


@app.get("/similar")
def similar(product_id: int = Query(..., description="Product ID"), top_k: int = 10):
    """Return similar items to a given product."""
    return {"product_id": product_id, "similar_items": similar_items(product_id, top_k)}
