import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, remove_stopwords, strip_short
from src.pipeline.data_processor import DataProcessor, CleanedData
from ..logging_config import setup_logging
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim

logger = setup_logging()


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ContentHandler:
    def __init__(self, user_listens_df, n_clusters=10):
        dp = DataProcessor()
        self.bookmarks_df = dp.load_table(CleanedData.BOOKMARKS)
        self.shiur_df = dp.load_table(CleanedData.SHIURIM)
        self.bookmarks_df = self.bookmarks_df.merge(self.shiur_df[['shiur', 'full_details']], on='shiur', how='inner')

        self.user_listens_df = user_listens_df
        self.user_listens_df['date'] = self.user_listens_df['date_played'].combine_first(
            self.user_listens_df['queue_date'])

        self.model = Word2Vec.load("./saved_models/content_filtering/word2vec_titles_v1.model")
        self.shiur_embeddings = self.get_shiur_embeddings(self.shiur_df)
        self.autoencoder = self.train_autoencoder()
        self.user_embeddings = self.get_user_embeddings()
        self.projected_shiur_embeddings = self.get_projected_shiur_embeddings()

        self.n_clusters = n_clusters
        self.cluster_labels = self.cluster_user_embeddings()

    def get_shiur_embeddings(self, shiur_df):
        shiur_df['embedding'] = shiur_df['full_details'].apply(self.get_title_vector)
        return shiur_df[['shiur', 'embedding']]

    def get_user_embeddings(self) -> pd.DataFrame:
        user_embeddings = {}
        for user, group in self.user_listens_df.groupby('user'):
            embeddings = [self.get_title_vector(details) for details in group['full_details']]
            embeddings_np = np.array(embeddings)  # Convert list of numpy arrays to a single numpy array

            embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32)
            user_embedding = self.autoencoder.encoder(embeddings_tensor)
            user_embedding = user_embedding.mean(dim=0).detach().numpy()  # Take the mean of the embeddings
            user_embeddings[user] = user_embedding

        user_embeddings_df = pd.DataFrame(list(user_embeddings.items()), columns=['user', 'embedding'])
        return user_embeddings_df

    def get_projected_shiur_embeddings(self) -> pd.DataFrame:
        shiur_embeddings = []
        for embedding in self.shiur_embeddings['embedding']:
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            projected_embedding = self.autoencoder.encoder(embedding_tensor).squeeze(0).detach().numpy()
            shiur_embeddings.append(projected_embedding)
        self.shiur_embeddings['projected_embedding'] = shiur_embeddings
        return self.shiur_embeddings[['shiur', 'projected_embedding']]

    def get_title_vector(self, title):
        lower_title = title.lower()
        custom_filters = [strip_punctuation, remove_stopwords, strip_short]
        processed_title = preprocess_string(lower_title, custom_filters)
        vectors = [self.model.wv[word] for word in processed_title if word in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.vector_size)

    def train_autoencoder(self, hidden_dim=64, epochs=50, learning_rate=1e-3):
        input_dim = self.model.vector_size
        autoencoder = Autoencoder(input_dim, hidden_dim)
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Prepare the data for training
        all_embeddings = []
        for user, group in self.user_listens_df.groupby('user'):
            embeddings = [self.get_title_vector(details) for details in group['full_details']]
            all_embeddings.extend(embeddings)

        all_embeddings = np.array(all_embeddings)
        train_tensor = torch.tensor(all_embeddings, dtype=torch.float32)

        # Train the autoencoder
        for epoch in range(epochs):
            autoencoder.train()
            optimizer.zero_grad()
            encoded, reconstructed = autoencoder(train_tensor)
            loss = criterion(reconstructed, train_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

        return autoencoder

    def cluster_user_embeddings(self):
        embeddings = np.stack(self.user_embeddings['embedding'].values)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.user_embeddings['cluster'] = kmeans.fit_predict(embeddings)
        return kmeans.labels_

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
            logger.warning(f"User {user_id} has no recent bookmark activity")
            return {}

        embedding = self.shiur_embeddings[self.shiur_embeddings['shiur'] == recent_shiur].iloc[0]['projected_embedding']
        similarities = cosine_similarity([embedding], np.stack(
            self.projected_shiur_embeddings['projected_embedding'].values)).flatten()
        similar_shiur_indices = similarities.argsort()[-top_n:][::-1][1:]
        similar_shiur_ids = self.projected_shiur_embeddings.iloc[similar_shiur_indices]['shiur'].values
        return {int(shiur_id): self.bookmarks_df[self.bookmarks_df['shiur'] == shiur_id]['full_details'].values[0] for shiur_id in similar_shiur_ids if not self.bookmarks_df[self.bookmarks_df['shiur'] == shiur_id].empty}

    def recommend_for_user_content(self, user_id: int, top_n: int = 5) -> Dict[int, str]:
        user_embeddings_filtered = self.user_embeddings[self.user_embeddings['user'] == user_id]
        if user_embeddings_filtered.empty:
            logger.warning(f"User {user_id} has no embeddings in the dataset")
            return {}

        user_embedding = user_embeddings_filtered.iloc[0]['embedding']
        user_cluster = user_embeddings_filtered.iloc[0]['cluster']

        # Find other users in the same cluster
        cluster_users = self.user_embeddings[self.user_embeddings['cluster'] == user_cluster]['user'].values

        # Get all the shiurim listened to by users in the same cluster
        cluster_shiurim = self.user_listens_df[self.user_listens_df['user'].isin(cluster_users)]['shiur'].unique()

        # Get the shiurim that the user has already listened to
        listened_shiurim = set(self.user_listens_df[self.user_listens_df['user'] == user_id]['shiur'].values)

        # Filter out the already listened shiurim from the cluster shiurim
        filtered_shiurim = [shiur for shiur in cluster_shiurim if shiur not in listened_shiurim]

        # Get embeddings for the filtered shiurim
        filtered_embeddings = self.projected_shiur_embeddings[self.projected_shiur_embeddings['shiur'].isin(
            filtered_shiurim)]
        all_shiur_vectors = np.stack(filtered_embeddings['projected_embedding'].values)

        similarities = cosine_similarity([user_embedding], all_shiur_vectors).flatten()

        # Get top_n recommendations
        similar_shiur_indices = similarities.argsort()[-top_n:][::-1]
        similar_shiur_ids = filtered_embeddings.iloc[similar_shiur_indices]['shiur'].values

        recommendations = {}
        for shiur_id in similar_shiur_ids:
            shiur_details = self.bookmarks_df[self.bookmarks_df['shiur'] == shiur_id]['full_details']
            if not shiur_details.empty:
                recommendations[int(shiur_id)] = shiur_details.values[0]
        return recommendations
