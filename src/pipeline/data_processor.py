import pandas as pd
from .db_connection import db_connection
from ..logging_config import setup_logging
import time

logger = setup_logging()


class DataProcessor:
    def __init__(self):
        self.db = db_connection()

    def load_df_csv(self, table: str) -> pd.DataFrame:
        logger.info(f"Loading data from: {table}.csv")
        return pd.read_csv(f"{table}.csv")

    def load_query(self, query: str) -> pd.DataFrame:
        logger.info(f"Loading data with query: {query}")
        return pd.read_sql(query, con=self.db)

    def save_to_db(self, df: pd.DataFrame, table_name: str):
        df.to_sql(table_name, con=self.db,
                  if_exists='replace', index=False)
        logger.info(f"Data saved to {table_name} table")

    def run_pipeline(self):
        from src.pipeline.etl import ETL
        from src.pipeline.data_preprocessing import DataPreprocessing

        etl = ETL()
        df_shiurim: pd.DataFrame = etl.get_shiurim_df()
        df_bookmarks: pd.DataFrame = etl.get_bookmarks_df()
        df_favorites: pd.DataFrame = etl.get_favorites_df()

        preprocessor = DataPreprocessing(
            df_shiurim, df_bookmarks, df_favorites)
        df_shiurim, df_bookmarks, df_favorites, df_categories = preprocessor.preprocess()

        df_shiurim.to_csv("shiurim_cleaned.csv")
        df_bookmarks.to_csv("bookmarks_cleaned.csv")
        df_favorites.to_csv("favorites_cleaned.csv")
        df_categories.to_csv("categories_cleaned.csv")
        self.save_to_db(df_shiurim, 'shiurim_cleaned')
        self.save_to_db(df_bookmarks, 'bookmarks_cleaned')
        self.save_to_db(df_favorites, 'favorites_cleaned')
        self.save_to_db(df_categories, 'categories_cleaned')


if __name__ == "__main__":
    processor = DataProcessor()
    start = time.time()
    processor.run_pipeline()
    end = time.time()
    length = round((end - start) / 60, 2)
    logger.info(f"Data Pipeline Complete: {length} min")
