import os
from typing import Dict
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, remove_stopwords, strip_short
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix, save_npz, load_npz
from sklearn.cluster import KMeans
from src.pipeline.data_processor import DataProcessor, CleanedData
from .base import BaseModel
from ..logging_config import setup_logging
from ..decorators import log_and_time_execution

logger = setup_logging()


class ContentFiltering(BaseModel):
    def __init__(self, k_clusters: int = 10):
        super().__init__()
        dp = DataProcessor()
        self.df = dp.load_table(CleanedData.SHIURIM)
        self.cat_df = dp.load_table(CleanedData.CATEGORIES)
        self.bookmarks_df = dp.load_table(CleanedData.BOOKMARKS)
        self.k_clusters = k_clusters
        self.__merge_cluster_info()

        self.user_listens_df = self.bookmarks_df.merge(
            self.shiur_df[['shiur', 'full_details']], on='shiur', how='inner')
        self.user_listens_df['date'] = self.user_listens_df['date_played'].combine_first(
            self.user_listens_df['queue_date'])

        self.model_path = "./saved_models/content_filtering/word2vec_titles_v1.model"
        self.similarity_matrix_directory = "./saved_models/content_filtering/sim_matrices"
        self.similarity_matrix_path = "./saved_models/content_filtering/sim_matrices/sim_matrix_cluster_{}.npz"

        self.similarity_matrices = {}

        if not os.path.exists(self.similarity_matrix_directory):
            os.makedirs(self.similarity_matrix_directory)

        if os.path.exists(self.model_path):
            self.model = Word2Vec.load(self.model_path)
            logger.info("Loaded Word2Vec Model")
        else:
            self.model = self.__train_word2vec_model()

        for cluster_id in range(self.k_clusters):
            path = self.similarity_matrix_path.format(cluster_id)
            if os.path.exists(path):
                similarity_matrix = self.__load_similarity_matrix(path)
                self.similarity_matrices[cluster_id] = self.__create_similarity_dataframe(
                    similarity_matrix, cluster_id)
                logger.info(
                    f"Loaded Similarity DataFrame for cluster {cluster_id}")
            else:
                self.__cluster_similarity_matrix(cluster_id)

    def get_recommendations(self, user_id: int, top_n: int = 5, *args, **kwargs) -> Dict[int, str]:
        shiur_id = self.get_most_recent_shiur(user_id)
        recommendations = self.get_weighted_recommendations(shiur_id, top_n)
        titles = self.df.set_index(
            'shiur').loc[recommendations.keys(), 'title']
        return {int(shiur_id): str(titles[shiur_id]) for shiur_id in recommendations.keys()}

    def get_weighted_recommendations(self, shiur_id: int, top_n: int = 5, *args, **kwargs) -> Dict[int, float]:
        cluster_id = self.df.loc[self.df['shiur']
                                 == shiur_id, 'Cluster'].values[0]
        similarity_scores = self.similarity_matrices[cluster_id].loc[shiur_id]
        most_similar_ids = similarity_scores.sort_values(
            ascending=False).index[1:top_n + 1]
        most_similar_scores = similarity_scores.sort_values(
            ascending=False).values[1:top_n + 1]

        recommendations = {int(shiur_id): float(score) for shiur_id, score in zip(
            most_similar_ids, most_similar_scores)}

        return recommendations

    @log_and_time_execution
    def __merge_cluster_info(self):
        # self.cat_df = self.cat_df.reset_index()
        if 'Cluster' not in self.cat_df.columns:
            self.__cluster_shiurim()
        self.df = self.df.merge(
            self.cat_df[['shiur', 'Cluster']], on='shiur', how='left')

    @log_and_time_execution
    def __cluster_shiurim(self):
        X = self.cat_df.iloc[:, 1:].values
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_
        self.cat_df['Cluster'] = labels

    @log_and_time_execution
    def __train_word2vec_model(self) -> Word2Vec:
        self.df['processed_title'] = self.df['full_details'].apply(
            self.__preprocess_title)

        model = Word2Vec(
            sentences=self.df['processed_title'],
            vector_size=200,  # Dimensionality of the word vectors
            window=5,  # Context window size
            min_count=5,  # Ignores all words with total frequency lower than this
            workers=4,  # Number of worker threads
            sg=1,  # Skip-gram model
            hs=0,  # Use negative sampling instead of hierarchical softmax
            negative=15,  # Number of negative samples
            epochs=20,  # Number of iterations over the corpus
            alpha=0.025,  # Initial learning rate
            min_alpha=0.0001,  # Final learning rate
            sample=1e-5  # Threshold for downsampling higher-frequency words
        )

        model.save(self.model_path)
        return model

    def __preprocess_title(self, title):
        lower_title = title.lower()
        custom_filters = [strip_punctuation, remove_stopwords, strip_short]
        return preprocess_string(lower_title, custom_filters)

    def __get_title_vector(self, title):
        processed_title = self.__preprocess_title(title)
        vectors = [self.model.wv[word]
                   for word in processed_title if word in self.model.wv]
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

    @log_and_time_execution
    def __cluster_similarity_matrix(self, cluster_id):
        cluster_df = self.df[self.df['Cluster'] == cluster_id]
        similarity_matrix = self.__compute_similarity_matrix(cluster_df)
        sim_matrix_path = self.similarity_matrix_path.format(cluster_id)
        self.__save_similarity_matrix(similarity_matrix, sim_matrix_path)
        self.similarity_matrices[cluster_id] = self.__create_similarity_dataframe(
            similarity_matrix, cluster_id)
        logger.info(
            f"Saved and created Similarity DataFrame for cluster {cluster_id}")

    def __compute_similarity_matrix(self, df: pd.DataFrame, threshold: float = 0.6, batch_size: int = 1500):
        n = len(df)
        similarity_matrix = lil_matrix((n, n))  # Using a sparse matrix

        vectors = np.stack(df['title'].apply(
            lambda x: self.__get_title_vector(x, self.model)).values)

        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            vectors_i = vectors[i:end_i]

            for j in range(i, n, batch_size):
                end_j = min(j + batch_size, n)
                vectors_j = vectors[j:end_j]

                cosine_sim = cosine_similarity(vectors_i, vectors_j)
                mask = cosine_sim > threshold

                similarity_matrix[i:end_i, j:end_j] = cosine_sim * mask

                if i != j:
                    similarity_matrix[j:end_j, i:end_i] = (cosine_sim * mask).T

        return similarity_matrix.tocsr()  # Convert to CSR format for efficient operations

    def __create_similarity_dataframe(self, similarity_matrix, cluster_id, chunk_size=1000):
        n = similarity_matrix.shape[0]
        ids = self.df[self.df['Cluster'] == cluster_id]['shiur'].values
        chunks = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_dense = similarity_matrix[start:end].toarray()
            chunk_df = pd.DataFrame(
                chunk_dense, index=ids[start:end], columns=ids)
            chunks.append(chunk_df)

        similarity_df = pd.concat(chunks, axis=0)
        return similarity_df

    def __save_similarity_matrix(self, matrix, file_path):
        save_npz(file_path, matrix)

    def __load_similarity_matrix(self, file_path):
        return load_npz(file_path)
