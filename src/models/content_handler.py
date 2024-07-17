import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any

from ..logging_config import setup_logging
from .content_models import ContentModel
from src.pipeline.data_processor import CleanedData

logger = setup_logging()


class ContentHandler:
    def __init__(self, bookmarks_shiur_df: pd.DataFrame = None, n_clusters: int = 10):
        """
        Initializes the ContentHandler.

        :param bookmarks_shiur_df: DataFrame containing bookmarks and shiur information. Used for testing/training sets.
        :param n_clusters: Number of clusters for user embeddings.
        """
        self.content_models = ContentModel()
        self.bookmarks_shiur_df = bookmarks_shiur_df if bookmarks_shiur_df is not None else self._get_bookmarks_shiur_df()
        self.bookmarks_shiur_df['date'] = self.bookmarks_shiur_df['date_played'].combine_first(
            self.bookmarks_shiur_df['queue_date'])
        self.n_clusters = n_clusters
        self.user_embeddings = self._get_projected_user_embeddings()
        self.shiur_embeddings = self._get_projected_shiur_embeddings()
        logger.info("ContentHandler instance created")

    def get_most_recent_shiur(self, user_id: int) -> Any:
        recent_listen = self.bookmarks_shiur_df.query(
            "user == @user_id & played == 1").sort_values("date", ascending=False)
        if not recent_listen.empty:
            return recent_listen.iloc[0]['shiur']
        recent_queue = self.bookmarks_shiur_df.query(
            "user == @user_id & bookmark == 'queue'").sort_values("date", ascending=False)
        return recent_queue.iloc[0]['shiur'] if not recent_queue.empty else None

    def recommend_based_on_recent_activity(self, user_id: int, top_n: int = 5) -> Dict[int, str]:
        recent_shiur = self.get_most_recent_shiur(user_id)
        if recent_shiur is None:
            logger.warning(f"User {user_id} has no bookmark activity")
            return {}

        embedding = self._get_shiur_embedding(recent_shiur)
        listened_shiurim = self.bookmarks_shiur_df.query("user == @user_id & played == 1")['shiur'].unique()
        other_shiurim = self.bookmarks_shiur_df[~self.bookmarks_shiur_df['shiur'].isin(
            listened_shiurim)]['shiur'].to_list()
        similar_shiur_ids = self._get_similar_shiur_ids(embedding, top_n, other_shiurim)
        return self._get_recommendations_from_ids(similar_shiur_ids)

    def recommend_for_user_content(self, user_id: int, top_n: int = 5) -> Dict[int, str]:
        user_embeddings_filtered = self.user_embeddings.query("user == @user_id")
        if user_embeddings_filtered.empty:
            logger.warning(f"User {user_id} has no embeddings in the dataset")
            return {}

        user_embedding = user_embeddings_filtered.iloc[0]['projected_embedding']
        user_cluster = user_embeddings_filtered.iloc[0]['cluster']
        # Gets all shiurim in the cluster that the user hasn't listened to
        cluster_shiurim = self._get_cluster_shiurim(user_cluster, user_id)
        # Gets the top_n most similar shiurim ids between the user_embedding and cluster_shiurim
        similar_shiur_ids = self._get_similar_shiur_ids(user_embedding, top_n, cluster_shiurim)
        # Returns details of the shiurim
        return self._get_recommendations_from_ids(similar_shiur_ids)

    def _get_bookmarks_shiur_df(self) -> pd.DataFrame:
        dp = self.content_models.dp
        shiur_df = self.content_models.shiur_df
        bookmarks_df = dp.load_table(CleanedData.BOOKMARKS)
        return bookmarks_df.merge(shiur_df[['shiur', 'title', 'full_details', 'embedding']], on='shiur', how='inner')

    def _get_projected_user_embeddings(self) -> pd.DataFrame:
        user_embeddings = {
            user: self._compute_mean_embedding(group['embedding'])
            for user, group in self.bookmarks_shiur_df.groupby('user')
        }
        user_embeddings = pd.DataFrame(user_embeddings.items(), columns=['user', 'projected_embedding'])
        self._cluster_user_embeddings(user_embeddings)
        return user_embeddings

    def _compute_mean_embedding(self, embeddings: pd.Series) -> np.ndarray:
        embeddings_tensor = torch.tensor(np.stack(embeddings), dtype=torch.float32)
        user_embedding = self.content_models.autoencoder.encoder(embeddings_tensor)
        return user_embedding.mean(dim=0).detach().numpy()

    def _cluster_user_embeddings(self, user_embeddings: pd.DataFrame) -> pd.DataFrame:
        embeddings = np.stack(user_embeddings['projected_embedding'].values)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        user_embeddings['cluster'] = kmeans.fit_predict(embeddings)
        return user_embeddings

    def _get_projected_shiur_embeddings(self) -> pd.DataFrame:
        embeddings = self.content_models.shiur_df['embedding'].apply(self._project_embedding)
        self.content_models.shiur_df['projected_embedding'] = embeddings
        return self.content_models.shiur_df[['shiur', 'projected_embedding']]

    def _project_embedding(self, embedding: Any) -> np.ndarray:
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        return self.content_models.autoencoder.encoder(embedding_tensor).squeeze(0).detach().numpy()

    def _get_shiur_embedding(self, shiur_id: int) -> np.ndarray:
        return self.shiur_embeddings.query("shiur == @shiur_id").iloc[0]['projected_embedding']

    def _get_cluster_shiurim(self, cluster: int, user_id: int) -> np.ndarray:
        cluster_users = self.user_embeddings.query("cluster == @cluster")['user'].values
        cluster_shiurim = self.bookmarks_shiur_df.query("user in @cluster_users")['shiur'].unique()
        listened_shiurim = self.bookmarks_shiur_df.query("user == @user_id")['shiur'].unique()
        return np.setdiff1d(cluster_shiurim, listened_shiurim)

    def _get_similar_shiur_ids(self, embedding: np.ndarray, top_n: int, shiur_ids: np.ndarray = None) -> np.ndarray:
        """
        Retrieves the IDs of shiurim most similar to a given embedding.

        :param embedding: Embedding to compare against.
        :param top_n: Number of similar shiurim to return.
        :param shiur_ids: Array of shiurim to compare against. Defaults to all shiurim.
        :return: Array of similar shiur IDs.
        """
        if shiur_ids is None or len(shiur_ids) == 0:
            filtered_embeddings = self.shiur_embeddings
        else:
            filtered_embeddings = self.shiur_embeddings.query("shiur in @shiur_ids")

        similarities = cosine_similarity([embedding], np.stack(
            filtered_embeddings['projected_embedding'].values)).flatten()
        similar_indices = similarities.argsort()[-top_n:][::-1]
        return filtered_embeddings.iloc[similar_indices]['shiur'].values

    def _get_recommendations_from_ids(self, sim_ids: np.ndarray) -> Dict[int, str]:
        recommendations = {}
        for shiur_id in sim_ids:
            shiur_details = self.bookmarks_shiur_df.query("shiur == @shiur_id")['full_details']
            if not shiur_details.empty:
                recommendations[int(shiur_id)] = shiur_details.values[0]
        return recommendations
