from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

# === Logging Setup ===
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# === Configuration ===
RAW_DATA_PATH = Path("../data/raw")
PROCESSED_DATA_PATH = Path("../data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

EVENTS_FILE = RAW_DATA_PATH / "events.csv"
ITEMS_PART1_FILE = RAW_DATA_PATH / "item_properties_part1.csv"
ITEMS_PART2_FILE = RAW_DATA_PATH / "item_properties_part2.csv"
CATEGORIES_FILE = RAW_DATA_PATH / "category_tree.csv"
OUTPUT_FILE = PROCESSED_DATA_PATH / "processed_sessions.parquet"

SESSION_TIMEOUT_SECONDS = 1800

def load_data():
    """
    Load the raw event and category datasets.
    Returns:
        events (pd.DataFrame): Raw user interaction logs.
        categories (pd.DataFrame): Item category mapping (currently unused).
    """
    logging.info("Loading events and category data...")
    events = pd.read_csv(EVENTS_FILE)
    categories = pd.read_csv(CATEGORIES_FILE)
    return events, categories


def preprocess_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and format the raw events data.
    Filters to relevant event types and parses timestamps.
    Args:
        events (pd.DataFrame): Raw events.
    Returns:
        pd.DataFrame: Cleaned and timestamped events.
    """
    logging.info("Filtering event types and converting timestamps...")
    events = events[events['event'].isin(['view', 'addtocart', 'transaction'])].copy()
    events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')
    return events


def process_item_properties(part1_file: Path, part2_file: Path) -> pd.DataFrame:
    """
    Load and merge item property files, keeping only the latest value per property.
    Args:
        part1_file (Path): Path to first item property file.
        part2_file (Path): Path to second item property file.
    Returns:
        pd.DataFrame: Wide-format item metadata.
    """
    logging.info("Loading item properties into memory...")
    df1 = pd.read_csv(part1_file)
    df2 = pd.read_csv(part2_file)
    items = pd.concat([df1, df2], ignore_index=True)
    items['timestamp'] = pd.to_datetime(items['timestamp'], unit='ms')

    logging.info("Selecting most recent item properties...")
    items.sort_values(by='timestamp', inplace=True)
    latest = items.drop_duplicates(subset=['itemid', 'property'], keep='last')
    item_features = latest.pivot(index='itemid', columns='property', values='value').reset_index()
    return item_features


def merge_and_sort(events: pd.DataFrame, item_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge item metadata into event logs and sort chronologically per user.
    Args:
        events (pd.DataFrame): Preprocessed events.
        item_features (pd.DataFrame): Metadata for each item.
    Returns:
        pd.DataFrame: Combined and sorted dataset.
    """
    logging.info("Merging entire event dataset with item features...")
    merged = events.merge(item_features, how='left', on='itemid')
    merged.sort_values(by=['visitorid', 'timestamp'], inplace=True)

    logging.info("Verifying required columns for saving...")
    expected_columns = ['session_id', 'visitorid', 'timestamp', 'itemid', 'event', 'categoryid']
    missing_columns = [col for col in expected_columns if col not in merged.columns]
    if missing_columns:
        logging.warning(f"Missing expected columns: {missing_columns}. Filling with NaN.")
        for col in missing_columns:
            merged[col] = np.nan

    return merged


def generate_sessions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Assign session IDs based on inactivity gaps between user interactions.
    Args:
        data (pd.DataFrame): Merged and sorted interaction log.
    Returns:
        pd.DataFrame: Same data with a new 'session_id' column.
    """
    logging.info("Generating session IDs...")
    data['prev_time'] = data.groupby('visitorid', group_keys=False)['timestamp'].shift(1)
    data['time_diff'] = (data['timestamp'] - data['prev_time']).dt.total_seconds()
    data['new_session'] = (data['time_diff'] > SESSION_TIMEOUT_SECONDS) | (data['time_diff'].isnull())
    data['session_id'] = data.groupby('visitorid', group_keys=False)['new_session'].cumsum().astype(np.int32)
    return data


def encode_categoricals(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Label encode specified categorical features to numeric format.
    Args:
        data (pd.DataFrame): Dataset with categorical columns.
        columns (list[str]): Column names to encode.
    Returns:
        pd.DataFrame: Encoded DataFrame.
    """
    logging.info("Encoding categorical features: %s", columns)
    for col in columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    return data


def save_data(data: pd.DataFrame, path: Path):
    """
    Save selected columns of the processed DataFrame to a Parquet file.
    Args:
        data (pd.DataFrame): Final dataset.
        path (Path): Output file path.
    """
    logging.info(f"Saving processed data to {path}...")
    selected_columns = ['session_id', 'visitorid', 'timestamp', 'itemid', 'event', 'categoryid']
    data[selected_columns].to_parquet(path)
    logging.info("✅ Preprocessing complete.")


def main():
    """
    Execute the full preprocessing pipeline:
    - Load data
    - Filter and process events
    - Extract item metadata
    - Merge and sort
    - Generate sessions
    - Encode categoricals
    - Save final dataset
    """
    logging.info("Starting preprocessing pipeline...")
    events, _ = load_data()
    events = preprocess_events(events)
    item_features = process_item_properties(ITEMS_PART1_FILE, ITEMS_PART2_FILE)
    data = merge_and_sort(events, item_features)
    data = generate_sessions(data)
    data = encode_categoricals(data, ['visitorid', 'itemid', 'categoryid'])
    save_data(data, OUTPUT_FILE)


if __name__ == "__main__":
    main()
