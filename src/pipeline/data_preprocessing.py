import pandas as pd
import numpy as np
from ..logging_config import setup_logging
from typing import Tuple

logger = setup_logging()


class DataPreprocessing:
    def __init__(self, df_shiurim: pd.DataFrame, df_bookmarks: pd.DataFrame, df_favorites: pd.DataFrame):
        self.df_shiurim = df_shiurim
        self.df_bookmarks = df_bookmarks
        self.df_favorites = df_favorites
        self.df_user_stats = pd.DataFrame()
        # One hot encoded matrix for all shiurim and their categories
        self.df_categories = pd.DataFrame()
        logger.info("DataPreprocessing instance created")

    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.__clean_data()

    def __clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data_cleaning_methods = {
            "Shiur Data": self.__clean_shiur_data,
            "Bookmark Data": self.__clean_bookmark_data,
            "Favorite Data": self.__clean_favorite_data,
            "User Statistics": self.__get_user_stats
        }

        for data_name, cleaning_method in data_cleaning_methods.items():
            logger.info(f"START: Cleaning {data_name}")
            cleaning_method()
            logger.info(f"FINISH: Cleaning {data_name}")

        return self.df_shiurim, self.df_bookmarks, self.df_favorites, self.df_categories, self.df_user_stats

    def __clean_shiur_data(self):
        # Subset specifies which fields can't be NaN
        self.df_shiurim.dropna(
            subset=['shiur', 'title', 'last_name', 'date', 'duration'], inplace=True)

        # Creates one hot encoding table for shiur and all categories
        self.__one_hot_cat()

        # This should be switched after mvp, for now we will remove duplicates from mult teachers/categories
        self.df_shiurim.drop_duplicates(subset=['shiur'], inplace=True)

        # Categories are ommitted from text cleaning as they are always formatted correctly
        text_columns = ['title', 'teacher_title', 'last_name',
                        'first_name', 'keywords', 'series_name', 'series_description']
        for col in text_columns:
            self.df_shiurim[col] = self.df_shiurim[col].apply(
                self.__clean_text)

        self.df_shiurim['duration'] = self.__convert_duration_to_seconds(
            self.df_shiurim['duration'])

        # This will be adjusted depending on needs during final iteration of content filtering
        self.df_shiurim['full_details'] = self.df_shiurim.apply(
            lambda row: f"Title {row['title']} Speaker {
                row['last_name']} Category {row['category']}",
            axis=1
        )

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

    def __clean_favorite_data(self):
        # No subset, all fields needed
        self.df_favorites.dropna(inplace=True)
        self.df_favorites.drop_duplicates(inplace=True)
        self.df_favorites['user'] = self.df_favorites['user'].astype(int)

    def __get_user_stats(self):
        shiur_stats_df = pd.DataFrame()
        listens_df = self.__get_listens(shiur_stats_df)
        downloads_df = self.__get_downloads(shiur_stats_df)
        teacher_df = self.__get_top_teacher(shiur_stats_df)
        self.df_user_stats = listens_df.merge(
            downloads_df, on='user').merge(teacher_df, on='user')
        self.df_user_stats = self.df_user_stats[[
            'total_listens', 'total_downloads', 'top_teacher']]

    def __get_top_teacher(self, shiur_stats_df: pd.DataFrame) -> pd.DataFrame:
        merged_df = pd.merge(self.df_bookmarks, self.df_shiurim, on='shiur')
        merged_df['name'] = merged_df['teacher_title'] + ' ' + \
            merged_df['first_name'] + ' ' + merged_df['last_name']
        merged_df.drop(
            columns=['teacher_title', 'last_name', 'first_name'], inplace=True)
        most_common_name = merged_df.groupby(
            ['user'])['name'].agg(pd.Series.mode)
        shiur_stats_df['top_teacher'] = most_common_name
        return shiur_stats_df

    def __get_downloads(self, shiur_stats_df: pd.DataFrame) -> pd.DataFrame:
        downloads_df = self.df_bookmarks[self.df_bookmarks['downloaded'] == 1].groupby(
            'user')
        shiur_stats_df['total_downloads'] = downloads_df['downloaded'].count()
        shiur_stats_df['total_downloads'].fillna(0)
        return shiur_stats_df

    def __get_listens(self, shiur_stats_df: pd.DataFrame) -> pd.DataFrame:
        listened_df = self.df_bookmarks[self.df_bookmarks['played'] == 1].groupby(
            'user')
        shiur_stats_df['total_listens'] = listened_df['played'].count()
        return shiur_stats_df

    def __one_hot_cat(self):
        df_categories = self.df_shiurim[[
            'shiur', 'category', 'middle_category', 'subcategory']].set_index('shiur')

        # One-hot encode 'category', 'middle_category', and 'subcategory' and combine them
        df_combined = pd.get_dummies(df_categories, columns=['category', 'middle_category', 'subcategory'],
                                     prefix=['category', 'middle_category', 'subcategory'], prefix_sep='_').astype(int)

        # Perform bitwise OR to combine the one-hot vectors for each 'shiur'
        df_combined = df_combined.groupby(
            'shiur').max().astype(int).sort_index(ascending=False)

        column_sums = df_combined.sum(axis=0)
        # All categories with less than 500 shiurim are grouped together to "Other"
        columns_to_aggregate = column_sums[column_sums < 500].index
        df_combined['Other'] = df_combined[columns_to_aggregate].max(axis=1)
        df_combined.drop(columns=columns_to_aggregate, inplace=True)

        # These two categories were causing conflicts in DB so they are combined into one column each
        if 'subcategory_Bein Adam L\'Chaveiro' and 'subcategory_Bein Adam l\'Chaveiro' in df_combined.columns:
            df_combined['subcategory_Bein Adam L\'Chaveiro'] = df_combined[[
                'subcategory_Bein Adam L\'Chaveiro', 'subcategory_Bein Adam l\'Chaveiro']].max(axis=1)
            df_combined.drop(
                columns=['subcategory_Bein Adam l\'Chaveiro'], inplace=True)
        if 'subcategory_Beit HaMikdash' and 'subcategory_Beit Hamikdash' in df_combined.columns:
            df_combined['subcategory_Beit HaMikdash'] = df_combined[[
                'subcategory_Beit HaMikdash', 'subcategory_Beit Hamikdash']].max(axis=1)
            df_combined.drop(
                columns=['subcategory_Beit Hamikdash'], inplace=True)

        self.df_categories = df_combined

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
