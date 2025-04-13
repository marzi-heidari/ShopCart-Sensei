# train_baselines.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import logging
from collections import defaultdict, Counter

from joblib import Parallel, delayed
from lightfm import LightFM
from lightfm.data import Dataset
from itertools import combinations

from tqdm import tqdm

# === Logging Setup ===
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# === Configuration ===
PROCESSED_DATA_PATH = Path("data/processed")
DATA_FILE = PROCESSED_DATA_PATH / "processed_sessions.parquet"
TOP_K = 10


class PopularityRecommender:
    """
    Recommends the most popular items globally based on their frequency in the dataset.
    This model does not personalize recommendations per user.
    """
    def __init__(self):
        self.popular_items = []

    def fit(self, df: pd.DataFrame):
        """
        Compute the most popular items by frequency.

        Args:
            df (pd.DataFrame): DataFrame containing an 'itemid' column.
        """
        logging.info("Fitting PopularityRecommender...")
        self.popular_items = df['itemid'].value_counts().nlargest(TOP_K).index.tolist()

    def recommend(self, user_id: int = None, k: int = TOP_K) -> List[int]:
        """
        Recommend top-k popular items.

        Args:
            user_id (int, optional): Not used. Returns same items for all users.
            k (int): Number of items to recommend.

        Returns:
            List[int]: List of item IDs.
        """
        return self.popular_items[:k]


def count_pairs(items):
    return [((i, j), 1) for i, j in combinations(sorted(items), 2)]


class ItemSimilarityRecommender:
    """
    Item-based collaborative filtering using co-occurrence within user sessions.
    Optimized using hash-based similarity accumulation and itertools combinations.
    """
    def __init__(self):
        self.similarity_matrix = defaultdict(lambda: defaultdict(int))

    def fit(self, df: pd.DataFrame):
        """
        Build item-item similarity matrix from sessions.

        Args:
            df (pd.DataFrame): Must include 'session_id' and 'itemid'.
        """
        logging.info("Fitting ItemSimilarityRecommender...")

        session_groups = df.groupby('session_id')['itemid'].apply(set)
        results = Parallel(n_jobs=-1)(delayed(count_pairs)(items) for items in session_groups)

        pair_counter = Counter()
        for pair_list in tqdm(results):
            pair_counter.update(pair_list)

        for (i, j), count in tqdm(pair_counter.items()):
            self.similarity_matrix[i][j] += count
            self.similarity_matrix[j][i] += count


def recommend(self, user_session: List[int], k: int = TOP_K) -> List[int]:
        """
        Recommend top-k items similar to a user session.

        Args:
            user_session (List[int]): List of item IDs interacted by user.
            k (int): Number of items to recommend.

        Returns:
            List[int]: List of item IDs.
        """
        scores = Counter()
        for item in user_session:
            for sim_item, score in self.similarity_matrix.get(item, {}).items():
                if sim_item not in user_session:
                    scores[sim_item] += score

        return [item for item, _ in scores.most_common(k)]

class MatrixFactorizationRecommender:
    """
    Matrix factorization-based recommender using LightFM with WARP loss.
    """
    def __init__(self):
        self.model = LightFM(loss='warp')
        self.dataset = Dataset()
        self.user_id_map = {}
        self.item_id_map = {}
        self.user_inv_map = {}
        self.item_inv_map = {}
        self.interactions = None

    def fit(self, df: pd.DataFrame):
        logging.info("Fitting MatrixFactorizationRecommender using LightFM...")
        users = df['visitorid'].astype(str)
        items = df['itemid'].astype(str)
        self.dataset.fit(users, items)
        self.user_id_map, self.user_inv_map = self.dataset.mapping()[0:2]
        self.item_id_map = self.dataset.mapping()[2]
        self.item_inv_map = {v: int(k) for k, v in self.item_id_map.items()}
        self.interactions, _ = self.dataset.build_interactions(zip(users, items))
        logging.info(f"Built interaction matrix: shape = {self.interactions.shape}")
        self.model.fit(self.interactions, epochs=5, num_threads=4)
        logging.info(f"Trained on {len(self.user_id_map)} users and {len(self.item_id_map)} items.")

    def recommend(self, user_id: int, k: int = TOP_K) -> List[int]:
        if str(user_id) not in self.user_id_map:
            logging.warning(f"User {user_id} not in user_id_map.")
            return []

        user_internal_id = self.user_id_map[str(user_id)]
        n_items = len(self.item_id_map)
        scores = self.model.predict(user_internal_id, np.arange(n_items), num_threads=4)
        logging.debug(f"Prediction scores shape: {scores.shape}, Sample: {scores[:5]}")

        top_items = np.argsort(-scores)[:k]
        logging.debug(f"Top internal item indices: {top_items}")

        recommendations = []
        for i in top_items:
            mapped = self.item_inv_map.get(int(i))
            if mapped is not None:
                recommendations.append(int(mapped))
            else:
                logging.warning(f"Missing item_inv_map for internal ID {i}")
        return recommendations


def load_data(path: Path) -> pd.DataFrame:
    """
    Load session-preprocessed data from Parquet.

    Args:
        path (Path): File location.

    Returns:
        pd.DataFrame: Loaded data.
    """
    logging.info(f"Loading data from {path}...")
    return pd.read_parquet(path)


def main():
    """
    Train and demonstrate baseline recommenders:
    - Popularity-based
    - Item-based collaborative filtering
    - Matrix Factorization
    """
    data = load_data(DATA_FILE)

    popularity_model = PopularityRecommender()
    popularity_model.fit(data)
    logging.info(f"Top-{TOP_K} Popular Items: {popularity_model.recommend()}")

    # item_cf_model = ItemSimilarityRecommender()
    # item_cf_model.fit(data)
    # session = data[data['visitorid'] == data['visitorid'].iloc[0]]['itemid'].tolist()
    # logging.info(f"Item-based CF Recommendations: {item_cf_model.recommend(session)}")

    mf_model = MatrixFactorizationRecommender()
    mf_model.fit(data)
    user_counts = data['visitorid'].value_counts()
    active_user = user_counts[user_counts >= 3].index[0]
    logging.info(f"Using active user {active_user} for matrix factorization test...")
    logging.info(f"Matrix Factorization Recommendations: {mf_model.recommend(active_user)}")


if __name__ == "__main__":
    main()
