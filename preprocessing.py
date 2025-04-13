from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# === Configuration ===
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

EVENTS_FILE = RAW_DATA_PATH / "events.csv"
ITEMS_PART1_FILE = RAW_DATA_PATH / "item_properties_part1.csv"
ITEMS_PART2_FILE = RAW_DATA_PATH / "item_properties_part2.csv"
CATEGORIES_FILE = RAW_DATA_PATH / "category_tree.csv"
OUTPUT_FILE = PROCESSED_DATA_PATH / "processed_sessions.parquet"

SESSION_TIMEOUT_SECONDS = 1800
CHUNK_SIZE = 50_000


def load_data():
    """
    Load the events and categories files. Item properties will be processed in chunks.
    Returns:
        Tuple of DataFrames: (events, categories)
    """
    events = pd.read_csv(EVENTS_FILE)
    categories = pd.read_csv(CATEGORIES_FILE)  # Not used yet
    return events, categories


def preprocess_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and format raw event data.
    Keeps only 'view', 'addtocart', 'transaction' events and converts timestamps.
    Args:
        events: Raw event DataFrame
    Returns:
        Cleaned DataFrame with parsed timestamps
    """
    events = events[events['event'].isin(['view', 'addtocart', 'transaction'])].copy()
    events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')
    return events


def process_item_chunks(file_paths: list[Path]) -> pd.DataFrame:
    """
    Process item property files in chunks and retain only latest property values.
    Args:
        file_paths: List of CSV Paths (parts 1 and 2)
    Returns:
        Aggregated and memory-efficient item features DataFrame
    """
    combined_chunks = []

    for path in file_paths:
        reader = pd.read_csv(path, chunksize=CHUNK_SIZE, usecols=['timestamp', 'itemid', 'property', 'value'])
        for chunk in reader:
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='ms')
            combined_chunks.append(chunk)

    all_items = pd.concat(combined_chunks, ignore_index=True)
    del combined_chunks  # Free memory early

    # Efficient sort and drop duplicates
    all_items.sort_values(by=['itemid', 'property', 'timestamp'], inplace=True)
    latest = all_items.drop_duplicates(subset=['itemid', 'property'], keep='last')
    item_features = latest.pivot(index='itemid', columns='property', values='value').reset_index()
    return item_features


def merge_and_sort(events: pd.DataFrame, item_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge event logs with item features and sort by visitor ID and timestamp.
    Args:
        events: Preprocessed event DataFrame
        item_features: Cleaned item features DataFrame
    Returns:
        Merged and sorted DataFrame
    """
    data = events.merge(item_features, how='left', on='itemid')
    data.sort_values(by=['visitorid', 'timestamp'], inplace=True)
    return data


def generate_sessions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Assign session IDs to user interactions using a 30-minute inactivity threshold.
    Args:
        data: Sorted interaction DataFrame
    Returns:
        DataFrame with new 'session_id' column
    """
    data['prev_time'] = data.groupby('visitorid', group_keys=False)['timestamp'].shift(1)
    data['time_diff'] = (data['timestamp'] - data['prev_time']).dt.total_seconds()
    data['new_session'] = (data['time_diff'] > SESSION_TIMEOUT_SECONDS) | (data['time_diff'].isnull())
    data['session_id'] = data.groupby('visitorid', group_keys=False)['new_session'].cumsum().astype(np.int32)
    return data


def encode_categoricals(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Label-encode selected categorical columns.
    Args:
        data: DataFrame with categorical columns
        columns: List of column names to encode
    Returns:
        DataFrame with encoded categorical columns
    """
    for col in columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    return data


def save_data(data: pd.DataFrame, path: Path):
    """
    Save processed data to a Parquet file with selected columns.
    Args:
        data: Final processed DataFrame
        path: Output file path
    """
    selected_columns = ['session_id', 'visitorid', 'timestamp', 'itemid', 'event', 'categoryid', 'brand']
    data[selected_columns].to_parquet(path)
    print(f"✅ Preprocessing complete. Saved to {path}")


def main():
    """
    End-to-end preprocessing pipeline with chunked and optimized item processing.
    """
    events, _ = load_data()
    events = preprocess_events(events)
    item_features = process_item_chunks([ITEMS_PART1_FILE, ITEMS_PART2_FILE])
    data = merge_and_sort(events, item_features)
    data = generate_sessions(data)
    data = encode_categoricals(data, ['visitorid', 'itemid', 'categoryid', 'brand'])
    save_data(data, OUTPUT_FILE)


if __name__ == "__main__":
    main()
