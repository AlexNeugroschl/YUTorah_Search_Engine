import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics.pairwise import cosine_similarity
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, remove_stopwords, strip_short
from src.pipeline.data_processor import DataProcessor, CleanedData
from ..logging_config import setup_logging
from gensim.models import Word2Vec
import torch
import torch.nn as nn

logger = setup_logging()


class Attention(nn.Module):
    def __init__(self, embed_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(embed_size, 1)

    def forward(self, embeddings):
        scores = self.attn(embeddings)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        weighted_embeddings = embeddings * attn_weights  # (batch_size, seq_len, embed_size)
        return torch.sum(weighted_embeddings, dim=1)  # (batch_size, embed_size)


class ContentHandler:
    def __init__(self, user_listens_df):
        dp = DataProcessor()
        self.bookmarks_df = dp.load_table(CleanedData.BOOKMARKS)
        self.shiur_df = dp.load_table(CleanedData.SHIURIM)
        self.bookmarks_df = self.bookmarks_df.merge(self.shiur_df[['shiur', 'full_details']], on='shiur', how='inner')

        self.user_listens_df = user_listens_df
        self.user_listens_df['date'] = self.user_listens_df['date_played'].combine_first(
            self.user_listens_df['queue_date'])

        self.model = Word2Vec.load("./saved_models/content_filtering/word2vec_titles_v1.model")
        self.shiur_embeddings = self.get_shiur_embeddings(self.shiur_df)
        self.attention = Attention(embed_size=self.model.vector_size)
        self.user_embeddings = self.get_user_embeddings()

    def get_shiur_embeddings(self, shiur_df):
        shiur_df['embedding'] = shiur_df['full_details'].apply(self.get_title_vector)
        return shiur_df[['shiur', 'embedding']]

    def get_user_embeddings(self) -> pd.DataFrame:
        user_embeddings = {}
        for user, group in self.user_listens_df.groupby('user'):
            embeddings = [self.get_title_vector(details) for details in group['full_details']]
            embeddings_np = np.array(embeddings)  # Convert list of numpy arrays to a single numpy array
            embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32).unsqueeze(
                0)  # (1, seq_len, embed_size)

            user_embedding = self.attention(embeddings_tensor).squeeze(0).detach().numpy()  # (embed_size)
            user_embeddings[user] = user_embedding

        user_embeddings_df = pd.DataFrame(list(user_embeddings.items()), columns=['user', 'embedding'])
        return user_embeddings_df

    def get_title_vector(self, title):
        lower_title = title.lower()
        custom_filters = [strip_punctuation, remove_stopwords, strip_short]
        processed_title = preprocess_string(lower_title, custom_filters)
        vectors = [self.model.wv[word] for word in processed_title if word in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.vector_size)

    def get_most_recent_shiur(self, user_id):
        recent_listen = self.user_listens_df[(self.user_listens_df['user'] == user_id)
                                             & (self.user_listens_df['played'] == 1)]
        if not recent_listen.empty:
            return recent_listen.sort_values("date", ascending=False).iloc[0]['shiur']

        recent_queue = self.user_listens_df[(self.user_listens_df['user'] == user_id)
                                            & (self.user_listens_df['bookmark'] == 'queue')]
        if not recent_queue.empty:
            return recent_queue.sort_values("queue_date", ascending=False).iloc[0]['shiur']

        return None

    def recommend_based_on_recent_activity(self, user_id, top_n: int = 5) -> Dict[int, str]:
        recent_shiur = self.get_most_recent_shiur(user_id)
        if recent_shiur is None:
            return ValueError(f"User {user_id} has no recent bookmark activity")

        embedding = self.shiur_embeddings[self.shiur_embeddings['shiur'] == recent_shiur].iloc[0]['embedding']
        similarities = cosine_similarity([embedding], np.stack(self.shiur_embeddings['embedding'].values)).flatten()
        similar_shiur_indices = similarities.argsort()[-top_n:][::-1][1:]
        similar_shiur_ids = self.shiur_embeddings.iloc[similar_shiur_indices]['shiur'].values
        return {int(shiur_id): self.bookmarks_df[self.bookmarks_df['shiur'] == shiur_id]['full_details'].values[0] for shiur_id in similar_shiur_ids if not self.bookmarks_df[self.bookmarks_df['shiur'] == shiur_id].empty}

    def recommend_for_user_content(self, user_id: int, top_n: int = 5) -> Dict[int, str]:
        user_embeddings_filtered = self.user_embeddings[self.user_embeddings['user'] == user_id]
        if user_embeddings_filtered.empty:
            logger.warning(f"User {user_id} has no embeddings in the dataset")
            return {}

        user_embedding = user_embeddings_filtered.iloc[0]['embedding']
        all_shiur_vectors = np.stack(self.shiur_embeddings['embedding'].values)
        similarities = cosine_similarity([user_embedding], all_shiur_vectors).flatten()

        # Get the shiurim that the user has already listened to
        listened_shiurim = set(self.user_listens_df[self.user_listens_df['user'] == user_id]['shiur'].values)

        # Filter out the already listened shiurim from the recommendations
        filtered_indices = []
        for idx in similarities.argsort()[::-1]:
            if self.shiur_embeddings.iloc[idx]['shiur'] not in listened_shiurim:
                filtered_indices.append(idx)
            if len(filtered_indices) >= top_n:
                break

        similar_shiur_ids = self.shiur_embeddings.iloc[filtered_indices]['shiur'].values

        recommendations = {}
        for shiur_id in similar_shiur_ids:
            shiur_details = self.bookmarks_df[self.bookmarks_df['shiur'] == shiur_id]['full_details']
            if not shiur_details.empty:
                recommendations[int(shiur_id)] = shiur_details.values[0]
        return recommendations
