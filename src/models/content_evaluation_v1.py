import numpy as np
import pandas as pd
from ..logging_config import setup_logging
from typing import List, Tuple

logger = setup_logging()


def hit_rate_at_k(recommended: List[int], relevant: List[int], k: int) -> Tuple[int, List[int]]:
    recommended_at_k = recommended[:k]
    hits = list(set(recommended_at_k).intersection(set(relevant)))
    return (1 if hits else 0), hits


def evaluate_hit_rate(test_df, handler, k=5):
    hits = 0
    test_users = test_df['user'].unique()
    total_users = len(test_users)
    logger.info(f"Total users in test set: {total_users}")
    user_hits = []

    for i, user in enumerate(test_users, start=1):
        relevant = test_df[test_df['user'] == user]['shiur'].values
        recommended = list(handler.recommend_for_user_content(user, top_n=k).keys())

        logger.info(f"Evaluating user {i}/{total_users} (ID: {user}):")
        logger.info(f"  Recommendations: {recommended}")
        logger.info(f"  Relevant items: {list(relevant)}")

        if len(recommended) > 0 and len(relevant) > 0:
            hit, hit_list = hit_rate_at_k(recommended, relevant, k)
            logger.info(f"  Hit: {hit}")
            if hit:
                hits += 1
                user_hits.append(user)
                logger.info(hit_list)
        else:
            logger.info(f"  No recommendations or no relevant items for user {user}")

    hit_rate = hits / total_users
    logger.info(f"Final hit rate: {hit_rate}")
    return hit_rate, user_hits


def split_user_interactions(df, test_size=0.2, interaction_threshold=5):
    train_data = []
    test_data = []

    for user, group in df.groupby('user'):
        interaction_count = len(group)
        if interaction_count > interaction_threshold:
            group = group.sort_values(by='date')
            P = int(len(group) * test_size)
            train_indices = group.index[:-P]
            test_indices = group.index[-P:]

            train_shiurs = group.loc[train_indices, 'shiur'].unique()
            test_shiurs = group.loc[test_indices, 'shiur'].unique()
            duplicates = np.intersect1d(train_shiurs, test_shiurs)

            if len(duplicates) > 0:
                for duplicate in duplicates:
                    # Assume to move the latest interaction to train
                    duplicate_index = group[group['shiur'] == duplicate].index[-1]
                    train_indices = train_indices.append(pd.Index([duplicate_index]))
                    test_indices = test_indices.drop(duplicate_index)

            train_data.append(group.loc[train_indices])
            test_data.append(group.loc[test_indices])
        else:
            train_data.append(group)

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)
    return train_df, test_df
