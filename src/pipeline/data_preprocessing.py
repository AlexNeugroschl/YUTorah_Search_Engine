import pandas as pd
from src.logging_config import setup_logging
from datetime import timedelta
from src.pipeline.db_connection import db_connection

logger = setup_logging()


class DataPreprocessing:
    def __init__(self, df_shiurim: pd.DataFrame, df_bookmarks: pd.DataFrame, df_favorites: pd.DataFrame):
        self.df_shiurim = df_shiurim
        self.df_bookmarks = df_bookmarks
        self.df_favorites = df_favorites
        self.df_categories = None
        logger.info("DataPreprocessing instance created")

    def preprocess(self) -> pd.DataFrame:
        return self.__clean_data()

    def __clean_data(self) -> pd.DataFrame:
        logger.info("START: Cleaning Data")

        self.__clean_shiur_data()
        self.__clean_boomark_data()
        self.__clean_favorite_data()

        logger.info("FINISHED: Cleaning Data")
        return self.df_shiurim, self.df_bookmarks, self.df_favorites, self.df_categories

    def __clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ''
        # Remove leading/trailing whitespaces
        text = text.strip()
        # Remove special characters
        text = ''.join(e for e in text if e.isalnum() or e.isspace())
        return text

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

        def convert_duration_to_seconds(duration_str):
            duration_datetime = pd.to_datetime(duration_str)
            duration_timedelta = timedelta(hours=duration_datetime.hour,
                                           minutes=duration_datetime.minute,
                                           seconds=duration_datetime.second,
                                           microseconds=duration_datetime.microsecond)
            return duration_timedelta.total_seconds()

        self.df_shiurim['duration'] = self.df_shiurim['duration'].apply(
            convert_duration_to_seconds)

        # This will be adjusted depending on needs during final iteration of content filtering
        self.df_shiurim['full_details'] = self.df_shiurim.apply(
            lambda row: f"Title {row['title']} Speaker {
                row['last_name']} Category {row['category']}",
            axis=1
        )

    def __clean_boomark_data(self) -> None:
        self.df_bookmarks.dropna(
            subset=['user', 'shiur', 'session'], inplace=True)
        self.df_bookmarks.drop_duplicates(inplace=True)
        self.df_bookmarks['user'] = self.df_bookmarks['user'].astype(int)

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
        df_combined_agg = df_combined.groupby('shiur').max()

        # Ensure values are 0 and 1 (not True/False)
        df_combined_agg = df_combined_agg.astype(int)

        # Sort by 'shiur' in descending order
        self.df_categories = df_combined_agg.sort_index(ascending=False)


if __name__ == "__main__":
    from src.pipeline.etl import ETL

    etl = ETL()
    df_shiurim: pd.DataFrame = etl.get_shiurim_df()
    df_bookmarks: pd.DataFrame = etl.get_bookmarks_df()
    df_favorites: pd.DataFrame = etl.get_favorites_df()

    preprocessor = DataPreprocessing(df_shiurim, df_bookmarks, df_favorites)
    df_shiurim, df_bookmarks, df_favorites, df_categories = preprocessor.preprocess()

    conn = db_connection()

    df_shiurim.to_csv("shiurim.csv")
    df_bookmarks.to_csv("bookmarks.csv")
    df_favorites.to_csv("favorites.csv")
    df_categories.to_csv("categories.csv")
