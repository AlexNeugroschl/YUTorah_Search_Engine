import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, remove_stopwords, strip_short
from typing import List

from ..decorators import log_and_time_execution
from ..logging_config import setup_logging
from src.pipeline.data_processor import DataProcessor, CleanedData

logger = setup_logging()


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ContentModel:
    WORD2VEC_PATH = "./saved_models/content_filtering/word2vec_titles_v1.model"
    AUTOENCODER_PATH = "./saved_models/content_filtering/autoencoder.model"
    VECTOR_SIZE = 200

    def __init__(self):
        self.dp = DataProcessor()
        self.shiur_df = self.dp.load_table(CleanedData.SHIURIM)
        self.word2vec = self._load_or_train_word2vec()
        self.shiur_df['embedding'] = self.shiur_df['full_details'].apply(self.get_title_vector)
        self.autoencoder = self._load_or_train_autoencoder()
        logger.info("ContentModel instance created")

    @log_and_time_execution
    def train_autoencoder(self, hidden_dim: int = 64, epochs: int = 60, learning_rate: float = 1e-3) -> Autoencoder:
        input_dim = self.word2vec.vector_size
        autoencoder = Autoencoder(input_dim, hidden_dim)
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        train_tensor = self.get_shiur_embeddings_tensor()

        for epoch in range(epochs):
            autoencoder.train()
            optimizer.zero_grad()
            encoded, reconstructed = autoencoder(train_tensor)
            loss = criterion(reconstructed, train_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 9:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        torch.save(autoencoder.state_dict(), self.AUTOENCODER_PATH)
        return autoencoder

    def load_autoencoder(self, hidden_dim: int = 64) -> Autoencoder:
        input_dim = self.VECTOR_SIZE
        autoencoder = Autoencoder(input_dim, hidden_dim)
        autoencoder.load_state_dict(torch.load(self.AUTOENCODER_PATH))
        autoencoder.eval()
        logger.info(f"Autoencoder model loaded from {self.AUTOENCODER_PATH}")
        return autoencoder

    @log_and_time_execution
    def train_word2vec(self) -> Word2Vec:
        model = Word2Vec(
            sentences=self.get_tokenized_titles(),
            vector_size=self.VECTOR_SIZE,
            window=5,
            min_count=5,
            workers=4,
            sg=1,
            hs=0,
            negative=15,
            epochs=20,
            alpha=0.025,
            min_alpha=0.0001,
            sample=1e-5
        )

        model.save(self.WORD2VEC_PATH)
        return model

    def _load_or_train_word2vec(self) -> Word2Vec:
        if os.path.exists(self.WORD2VEC_PATH):
            return Word2Vec.load(self.WORD2VEC_PATH)
        return self.train_word2vec()

    def _load_or_train_autoencoder(self) -> Autoencoder:
        if os.path.exists(self.AUTOENCODER_PATH):
            return self.load_autoencoder()
        return self.train_autoencoder()

    def get_tokenized_titles(self) -> pd.Series:
        return self.shiur_df['full_details'].apply(self.get_preprocess_title)

    def get_preprocess_title(self, title: str) -> List[str]:
        custom_filters = [strip_punctuation, remove_stopwords, strip_short]
        return preprocess_string(title.lower(), custom_filters)

    def get_title_vector(self, title: str) -> np.ndarray:
        processed_title = self.get_preprocess_title(title)
        vectors = [self.word2vec.wv[word] for word in processed_title if word in self.word2vec.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.word2vec.vector_size)

    def get_shiur_embeddings_tensor(self) -> torch.Tensor:
        all_embeddings = np.array(self.shiur_df['embedding'].to_list())
        return torch.tensor(all_embeddings, dtype=torch.float32)
