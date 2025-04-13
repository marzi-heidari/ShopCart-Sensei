# preprocessing.py

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# === Configuration ===
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

EVENTS_FILE = RAW_DATA_PATH / "events.csv"
ITEMS_FILE = RAW_DATA_PATH / "item_properties.csv"
CATEGORIES_FILE = RAW_DATA_PATH / "category_tree.csv"
OUTPUT_FILE = PROCESSED_DATA_PATH / "processed_sessions.parquet"

SESSION_TIMEOUT_SECONDS = 1800


def load_data():
    events = pd.read_csv(EVENTS_FILE)
    items = pd.read_csv(ITEMS_FILE)
    categories = pd.read_csv(CATEGORIES_FILE)  # Not used yet
    return events, items, categories


def preprocess_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events[events['event'].isin(['view', 'addtocart', 'transaction'])].copy()
    events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')
    return events


def preprocess_item_features(items: pd.DataFrame) -> pd.DataFrame:
    items['timestamp'] = pd.to_datetime(items['timestamp'], unit='ms')
    items = items.sort_values(by='timestamp').drop_duplicates(['itemid', 'property'], keep='last')
    item_features = items.pivot(index='itemid', columns='property', values='value').reset_index()
    return item_features


def merge_and_sort(events: pd.DataFrame, item_features: pd.DataFrame) -> pd.DataFrame:
    data = events.merge(item_features, how='left', on='itemid')
    data = data.sort_values(by=['visitorid', 'timestamp']).reset_index(drop=True)
    return data


def generate_sessions(data: pd.DataFrame) -> pd.DataFrame:
    data['prev_time'] = data.groupby('visitorid')['timestamp'].shift(1)
    data['time_diff'] = (data['timestamp'] - data['prev_time']).dt.total_seconds()
    data['new_session'] = (data['time_diff'] > SESSION_TIMEOUT_SECONDS) | (data['time_diff'].isnull())
    data['session_id'] = data.groupby('visitorid')['new_session'].cumsum()
    return data


def encode_categoricals(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    return data


def save_data(data: pd.DataFrame, path: Path):
    selected_columns = ['session_id', 'visitorid', 'timestamp', 'itemid', 'event', 'categoryid', 'brand']
    data[selected_columns].to_parquet(path)
    print(f"✅ Preprocessing complete. Saved to {path}")


def main():
    events, items, _ = load_data()
    events = preprocess_events(events)
    item_features = preprocess_item_features(items)
    data = merge_and_sort(events, item_features)
    data = generate_sessions(data)
    data = encode_categoricals(data, ['visitorid', 'itemid', 'categoryid', 'brand'])
    save_data(data, OUTPUT_FILE)


if __name__ == "__main__":
    main()