import pandas as pd
import numpy as np
from src.logging_config import setup_logging
from typing import Tuple

logger = setup_logging()


class DataPreprocessing:
    def __init__(self, df_shiurim: pd.DataFrame, df_bookmarks: pd.DataFrame, df_favorites: pd.DataFrame):
        self.df_shiurim = df_shiurim
        self.df_bookmarks = df_bookmarks
        self.df_favorites = df_favorites
        # One hot encoded matrix for all shiurim and their categories
        self.df_categories: pd.DataFrame = None
        logger.info("DataPreprocessing instance created")

    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.__clean_data()

    def __clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data_cleaning_methods = {
            "Shiur Data": self.__clean_shiur_data,
            "Bookmark Data": self.__clean_bookmark_data,
            "Favorite Data": self.__clean_favorite_data
        }

        for data_name, cleaning_method in data_cleaning_methods.items():
            logger.info(f"START: Cleaning {data_name}")
            cleaning_method()
            logger.info(f"FINISH: Cleaning {data_name}")

        return self.df_shiurim, self.df_bookmarks, self.df_favorites, self.df_categories

    def __clean_shiur_data(self) -> None:
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

    def __clean_bookmark_data(self) -> None:
        self.df_bookmarks.dropna(
            subset=['user', 'shiur', 'session', 'duration'], inplace=True)

        self.df_bookmarks.drop_duplicates(inplace=True)

        self.df_bookmarks['user'] = self.df_bookmarks['user'].astype(int)

        self.df_bookmarks['timestamp'] = self.df_bookmarks['timestamp'].fillna(
            0)

        self.df_bookmarks['duration'] = self.__convert_duration_to_seconds(
            self.df_bookmarks['duration'])

        self.__listen_percentage_chunks()

    def __listen_percentage_chunks(self, chunk_size: int = 500_000) -> None:
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

    def __clean_favorite_data(self) -> None:
        # No subset, all fields needed
        self.df_favorites.dropna(inplace=True)
        self.df_favorites.drop_duplicates(inplace=True)
        self.df_favorites['user'] = self.df_favorites['user'].astype(int)

    def __one_hot_cat(self) -> None:
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


if __name__ == "__main__":
    from src.pipeline.etl import ETL

    etl = ETL()
    df_shiurim: pd.DataFrame = etl.get_shiurim_df()
    df_bookmarks: pd.DataFrame = etl.get_bookmarks_df()
    df_favorites: pd.DataFrame = etl.get_favorites_df()

    preprocessor = DataPreprocessing(df_shiurim, df_bookmarks, df_favorites)
    df_shiurim, df_bookmarks, df_favorites, df_categories = preprocessor.preprocess()

    df_shiurim.to_csv("shiurim.csv")
    df_bookmarks.to_csv("bookmarks.csv")
    df_favorites.to_csv("favorites.csv")
    df_categories.to_csv("categories.csv")
