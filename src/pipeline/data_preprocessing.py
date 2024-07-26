import pandas as pd
import numpy as np
from typing import Tuple
from ..logging_config import setup_logging
from ..decorators import log_and_time_execution

logger = setup_logging()


class DataPreprocessing:
    def __init__(self, df_shiurim: pd.DataFrame, df_bookmarks: pd.DataFrame, df_favorites: pd.DataFrame):
        self.df_shiurim = df_shiurim
        self.df_bookmarks = df_bookmarks
        self.df_favorites = df_favorites
        # One hot encoded matrix for all shiurim and their categories
        self.df_categories = pd.DataFrame()
        logger.info("DataPreprocessing instance created")

    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.__clean_data()

    def __clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.__clean_shiur_data()
        self.__clean_bookmark_data()
        self.__clean_favorite_data()
        return self.df_shiurim, self.df_bookmarks, self.df_favorites, self.df_categories

    @log_and_time_execution
    def __clean_shiur_data(self):
        # Subset specifies which fields can't be NaN
        self.df_shiurim.dropna(
            subset=['shiur', 'title', 'last_name', 'date', 'duration'], inplace=True)

        # Creates one hot encoding table for shiur and all categories
        self.__one_hot_cat()

        # This should be switched after mvp, for now we will remove duplicates from mult teachers/categories
        self.df_shiurim.drop_duplicates(subset=['shiur'], inplace=True)

        # Categories are ommitted from text cleaning as they are always formatted correctly
        text_columns = ['teacher_title', 'last_name',
                        'first_name', 'keywords', 'series_name', 'series_description']
        for col in text_columns:
            self.df_shiurim[col] = self.df_shiurim[col].apply(
                self.__clean_text)

        self.df_shiurim['duration'] = self.__convert_duration_to_seconds(
            self.df_shiurim['duration'])

        # This will be adjusted depending on needs during final iteration of content filtering
        self.df_shiurim['full_details'] = self.df_shiurim.apply(
            lambda row: f"Title {row['title']} Speaker {row['last_name']} Category {row['category']}", axis=1)
        
        self.df_shiurim['full_details'] = self.df_shiurim['full_details'].apply(
            self.__clean_text)

    @log_and_time_execution
    def __clean_bookmark_data(self):
        self.df_bookmarks.dropna(
            subset=['user', 'shiur', 'session', 'duration'], inplace=True)

        self.df_bookmarks.drop_duplicates(inplace=True)

        self.df_bookmarks['user'] = self.df_bookmarks['user'].astype(int)

        self.df_bookmarks['timestamp'] = self.df_bookmarks['timestamp'].fillna(
            0)

        self.df_bookmarks['duration'] = self.__convert_duration_to_seconds(
            self.df_bookmarks['duration'])

        self.__listen_percentage_chunks()

    def __listen_percentage_chunks(self, chunk_size: int = 500_000):
        num_chunks = max(1, len(self.df_bookmarks) // chunk_size + 1)
        listen_percentage = []

        for i in range(num_chunks):
            chunk = self.df_bookmarks.iloc[i * chunk_size:(i + 1) * chunk_size]

            chunk_listen_percentage = np.where(
                chunk['duration'] != 0,
                chunk['timestamp'] / chunk['duration'],
                0
            )

            chunk_listen_percentage = np.round(chunk_listen_percentage, 3)

            listen_percentage.append(chunk_listen_percentage)

        self.df_bookmarks['listen_percentage'] = np.concatenate(
            listen_percentage)

    @log_and_time_execution
    def __clean_favorite_data(self):
        # No subset, all fields needed
        self.df_favorites.dropna(inplace=True)
        self.df_favorites.drop_duplicates(inplace=True)
        self.df_favorites['user'] = self.df_favorites['user'].astype(int)

    def __one_hot_cat(self):
        df_categories = self.df_shiurim[[
            'shiur', 'category', 'middle_category', 'subcategory']].set_index('shiur')

        # One-hot encode 'category', 'middle_category', and 'subcategory' and combine them
        df_combined = pd.get_dummies(df_categories, columns=['category', 'middle_category', 'subcategory'],
                                     prefix=['category', 'middle_category', 'subcategory'], prefix_sep='_').astype(int)

        # Perform bitwise OR to combine the one-hot vectors for each 'shiur'
        df_combined = df_combined.groupby(
            'shiur').max().astype(int).sort_index(ascending=False)

        col_pairs = [
            ('subcategory_Bein Adam L\'Chaveiro', 'subcategory_Bein Adam l\'Chaveiro'), 
            ('subcategory_Beit HaMikdash', 'subcategory_Beit Hamikdash'),
            ('subcategory_Berachos', 'subcategory_Berachot'),
            ('subcategory_Berachos', 'subcategory_Brachot'),
            ('subcategory_Peah', 'subcategory_Pe\'ah'),
            ('subcategory_Terumos', 'subcategory_Terumot'),
            ('subcategory_Maaser Sheni', 'subcategory_Ma\'aser Sheni'),
            ('subcategory_Challah', 'subcategory_Chala'),
            ('subcategory_Maaseros', 'subcategory_Ma\'asrot'),
            ('subcategory_Orla', 'subcategory_Orlah'),
            ('subcategory_Sheviis', 'subcategory_Shevi\'it'),
            ('subcategory_Shabbos', 'subcategory_Shabbat'),
            ('subcategory_Yuma', 'subcategory_Yoma'),
            ('subcategory_Chagiga', 'subcategory_Chagigah'),
            ('subcategory_Moed Katan', 'subcategory_Moed Kattan'),
            ('subcategory_Taanis', 'subcategory_Ta\'anit'),
            ('subcategory_Taanis', 'subcategory_Taanit'),
            ('subcategory_Yevamos', 'subcategory_Yevamot'),
            ('subcategory_Kesubos', 'subcategory_Ketuvot'),
            ('subcategory_Bava Basra', 'subcategory_Bava Batra'),
            ('subcategory_Shavuos', 'subcategory_Shevuot'),
            ('subcategory_Makkos', 'subcategory_Makkot'),
            ('subcategory_Makkos', 'subcategory_Makot'),
            ('subcategory_Avoda Zara', 'subcategory_Avodah Zara'),
            ('subcategory_Horayos', 'subcategory_Horayot'),
            ('subcategory_Nidah', 'subcategory_Niddah')
            ]
        for col1, col2 in col_pairs:
            df_combined[col1] = df_combined[[col1, col2]].max(axis=1)
            df_combined.drop(columns=[col2], inplace=True)
        df_combined['subcategory_Tazria-Metzora'] = df_combined[['subcategory_Tazria', 'subcategory_Metzora']].max(axis=1)
        df_combined['subcategory_Nitzavim-Vayeilech'] = df_combined[['subcategory_Nitzavim', 'subcategory_Vayeilech']].max(axis=1)

        self.df_categories = df_combined.reset_index()

    def __clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ''
        return ''.join(e for e in text.strip() if e.isalnum() or e.isspace())

    def __convert_duration_to_seconds(self, duration_series: pd.Series) -> pd.Series:
        # Extract the time component from the datetime string
        time_strs = duration_series.str.split().str[1]
        # Split the time component into hours, minutes, seconds, and milliseconds
        time_parts = time_strs.str.split(':', expand=True)
        seconds_parts = time_parts[2].str.split('.', expand=True)
        time_parts[2] = seconds_parts[0]

        # Convert to total seconds
        total_seconds = (
            time_parts[0].astype(float) * 3600 +
            time_parts[1].astype(float) * 60 +
            time_parts[2].astype(float)
        )
        return total_seconds
