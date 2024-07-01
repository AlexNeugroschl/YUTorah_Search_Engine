import pandas as pd
from src.logging_config import setup_logging
from typing import Tuple
from datetime import timedelta

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
        logger.info("START: Cleaning Data")

        self.__clean_shiur_data()
        self.__clean_bookmark_data()
        self.__clean_favorite_data()

        logger.info("FINISHED: Cleaning Data")
        return self.df_shiurim, self.df_bookmarks, self.df_favorites, self.df_categories

    def __clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ''
        return ''.join(e for e in text.strip() if e.isalnum() or e.isspace())

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

        self.df_shiurim['duration'] = self.df_shiurim['duration'].apply(
            self.__convert_duration_to_seconds)

        # This will be adjusted depending on needs during final iteration of content filtering
        self.df_shiurim['full_details'] = self.df_shiurim.apply(
            lambda row: f"Title {row['title']} Speaker {
                row['last_name']} Category {row['category']}",
            axis=1
        )

    def __convert_duration_to_seconds(self, duration_str: str) -> float:
        duration_datetime = pd.to_datetime(duration_str)
        duration_timedelta = timedelta(hours=duration_datetime.hour,
                                       minutes=duration_datetime.minute,
                                       seconds=duration_datetime.second,
                                       microseconds=duration_datetime.microsecond)
        return duration_timedelta.total_seconds()

    def __clean_bookmark_data(self) -> None:
        self.df_bookmarks.dropna(
            subset=['user', 'shiur', 'session', 'duration'], inplace=True)

        self.df_bookmarks.drop_duplicates(inplace=True)

        self.df_bookmarks['user'] = self.df_bookmarks['user'].astype(int)

        self.df_bookmarks['timestamp'] = self.df_bookmarks['timestamp'].fillna(
            0)

        self.df_bookmarks['duration'] = self.df_bookmarks['duration'].apply(
            self.__convert_duration_to_seconds)

        self.df_bookmarks['listen_percentage'] = self.df_bookmarks.apply(
            lambda row: round(row['timestamp'] / row['duration'], 3) if row['duration'] != 0 else 0, axis=1)

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
        self.df_categories = df_combined.groupby(
            'shiur').max().astype(int).sort_index(ascending=False)


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
