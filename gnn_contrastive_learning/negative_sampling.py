import pickle
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dataset_classes.spider_dataset import SpiderDataset
from sentence_transformers import SentenceTransformer

def save_embeddings(embeddings, save_path="question_embeddings.pkl"):
    """
    Save precomputed question embeddings to disk.
    :param embeddings: Dictionary of question embeddings.
    :param save_path: File path for saving the embeddings.
    """
    with open(save_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {save_path}")

def load_embeddings(load_path="question_embeddings.pkl"):
    """
    Load precomputed question embeddings from disk.
    :param load_path: File path for loading the embeddings.
    :return: Dictionary of question embeddings.
    """
    with open(load_path, "rb") as f:
        embeddings = pickle.load(f)
    print(f"Embeddings loaded from {load_path}")
    return embeddings

def filter_by_database(anchor_db_id, records):
    """
    Exclude all instances from the same database as the anchor.
    :param anchor_db_id: Database ID of the anchor instance.
    :param dataset: List of dataset records, each containing 'db_id' and 'question'.
    :return: Filtered dataset indices excluding instances from the same database.
    """
    return [
        idx for idx, record in enumerate(records)
        if record['db_id'] != anchor_db_id
    ]

def select_negative_instances(
    anchor_idx,
    records,
    question_embeddings,
    easy_start=0.0,
    easy_end=0.3,
    hard_start=0.3,
    hard_end=0.6,
    hard_ratio=0.4,
    num_negatives=5,
):
    """
    Select negative instances with configurable ranges for easy and hard negatives.
    :param anchor_idx: Index of the anchor instance.
    :param records: List of dataset records.
    :param question_embeddings: Precomputed embeddings of all questions.
    :param easy_start: Start percentile for easy negatives (e.g., 0.0).
    :param easy_end: End percentile for easy negatives (e.g., 0.2).
    :param hard_start: Start percentile for hard negatives (e.g., 0.2).
    :param hard_end: End percentile for hard negatives (e.g., 0.5).
    :param hard_ratio: Fraction of negatives to select as hard negatives.
    :param num_negatives: Total number of negatives to select.
    :return: Indices of selected negative instances.
    """
    assert len(records) == len(question_embeddings), "Mismatch between records and embeddings."
    anchor_db_id = records[anchor_idx]["db_id"]

    # Step 1: Filter instances from other databases
    valid_indices = filter_by_database(anchor_db_id, records)
    if not valid_indices:
        print("No valid negatives available from other databases.")
        return []

    # Step 2: Compute similarities
    anchor_embedding = question_embeddings[anchor_idx].reshape(1, -1)
    similarities = cosine_similarity(anchor_embedding, question_embeddings[valid_indices])[0]

    # Step 3: Sort indices by similarity
    sorted_indices = np.argsort(similarities)  # Least similar to most similar

    # Step 4: Define ranges for easy and hard negatives
    total_valid = len(sorted_indices)
    easy_range = sorted_indices[
        int(total_valid * easy_start) : int(total_valid * easy_end)
    ]
    hard_range = sorted_indices[
        int(total_valid * hard_start) : int(total_valid * hard_end)
    ]

    # Step 5: Sample negatives
    num_hard_negatives = int(num_negatives * hard_ratio)
    num_easy_negatives = num_negatives - num_hard_negatives

    # Sample from ranges
    sampled_hard_negatives = np.random.choice(
        hard_range, size=min(num_hard_negatives, len(hard_range)), replace=False
    )
    sampled_easy_negatives = np.random.choice(
        easy_range, size=min(num_easy_negatives, len(easy_range)), replace=False
    )

    # Combine and map back to original indices
    all_sampled_indices = np.concatenate([sampled_hard_negatives, sampled_easy_negatives])
    return [valid_indices[idx] for idx in all_sampled_indices]

