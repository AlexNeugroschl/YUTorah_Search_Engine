import os
from typing import Dict, List
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, remove_stopwords, strip_short
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix, save_npz, load_npz
from src.pipeline.data_processor import DataProcessor, CleanedData
from .base import BaseModel
from ..logging_config import setup_logging
from ..decorators import log_and_time_execution


logger = setup_logging()


class ContentFiltering(BaseModel):
    def __init__(self):
        super().__init__()
        dp = DataProcessor()
        self.df = dp.load_limit_table(CleanedData.SHIURIM, 70_000)

        self.model_path = "./saved_models/word2vec_titles_v1.model"
        self.similarity_matrix_path = "./saved_models/shiur_similarity_matrix_v1.npz"

        if os.path.exists(self.similarity_matrix_path):
            similarity_matrix = self.__load_similarity_matrix(
                self.similarity_matrix_path)
            self.similarity_df = self.__create_similarity_dataframe(
                similarity_matrix)
            logger.info("Loaded Similarity DataFrame")
        else:
            if os.path.exists(self.model_path):
                self.model = Word2Vec.load(self.model_path)
                logger.info("Loaded Word2Vec Model")
            else:
                self.model = self.__train_word2vec_model()

            similarity_matrix = self.__compute_similarity_matrix()
            self.__save_similarity_matrix(
                similarity_matrix, self.similarity_matrix_path)
            self.similarity_df = self.__create_similarity_dataframe(
                similarity_matrix)

    def get_recommendations(self, shiur_id: int, top_n: int = 5, *args, **kwargs) -> Dict[int, str]:
        recommendations = self.get_weighted_recommendations(shiur_id, top_n)
        titles = self.df.set_index(
            'shiur').loc[recommendations.keys(), 'title']
        return {int(shiur_id): str(titles[shiur_id]) for shiur_id in recommendations.keys()}

    def get_weighted_recommendations(self, shiur_id: str, top_n: int = 5, *args, **kwargs) -> Dict[int, float]:
        similarity_scores = self.similarity_df.loc[shiur_id]
        most_similar_ids = similarity_scores.sort_values(
            ascending=False).index[1:top_n + 1]
        most_similar_scores = similarity_scores.sort_values(
            ascending=False).values[1:top_n + 1]

        recommendations = {int(shiur_id): float(score) for shiur_id, score in zip(
            most_similar_ids, most_similar_scores)}

        return recommendations

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

    def __get_title_vector(self, title, model):
        processed_title = self.__preprocess_title(title)
        vectors = [model.wv[word]
                   for word in processed_title if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    @log_and_time_execution
    def __compute_similarity_matrix(self, threshold=0.6, batch_size=1000):
        n = len(self.df)
        similarity_matrix = lil_matrix((n, n))  # Using a sparse matrix

        vectors = np.stack(self.df['title'].apply(
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

    @log_and_time_execution
    def __create_similarity_dataframe(self, similarity_matrix, chunk_size=1000):
        n = similarity_matrix.shape[0]
        ids = self.df['shiur'].values
        chunks = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_dense = similarity_matrix[start:end].toarray()
            chunk_df = pd.DataFrame(
                chunk_dense, index=ids[start:end], columns=ids)
            chunks.append(chunk_df)

        similarity_df = pd.concat(chunks, axis=0)
        return similarity_df

    @log_and_time_execution
    def __save_similarity_matrix(self, matrix, file_path):
        save_npz(file_path, matrix)

    def __load_similarity_matrix(self, file_path):
        return load_npz(file_path)
