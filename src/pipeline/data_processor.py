import time
import pandas as pd
from enum import Enum
from .user_taste import UserTaste
from .db_connection import db_connection
from ..logging_config import setup_logging
from .calendar_generator import generate_calendar


logger = setup_logging()


class CleanedData(Enum):
    SHIURIM = "shiurim_cleaned"
    BOOKMARKS = "bookmarks_cleaned"
    FAVORITES = "favorites_cleaned"
    CATEGORIES = "categories_cleaned"
    USER_TASTE = "user_taste_cleaned"
    CALENDAR = "cycles_calendar"

class DataProcessor:
    def __init__(self):
        self.db = db_connection()
        logger.info("DataProcessor instance created")

    def load_table(self, table_name: str) -> pd.DataFrame:
        logger.info(f"Loading data from: {table_name.value}")
        query = f"SELECT * FROM {table_name.value}"
        return pd.read_sql(query, con=self.db, parse_dates=self.__get_date_columns(table_name))

    def load_limit_table(self, table_name: str, entries: int = 100_000) -> pd.DataFrame:
        logger.info(f"Loading {entries} entries from: {table_name.value}")
        query = f"SELECT * FROM {table_name.value} DESC LIMIT {entries}"
        return pd.read_sql_query(query, con=self.db, parse_dates=self.__get_date_columns(table_name))

    def load_query(self, query: str) -> pd.DataFrame:
        logger.info(f"Loading data with query: {query}")
        return pd.read_sql(query, con=self.db)

    def __save_to_db(self, df: pd.DataFrame, table_name: str):
        df.to_sql(table_name.value, con=self.db,
                  if_exists='replace', index=False)
        logger.info(f"Data saved to {table_name.value} table")

    def __get_date_columns(self, table_name: str) -> list:
        if table_name == CleanedData.SHIURIM:
            return ['date']
        elif table_name == CleanedData.BOOKMARKS:
            return ['date_played', 'date_downloaded', 'queue_date']
        elif table_name == CleanedData.FAVORITES:
            return ['date_favorite_added']
        else:
            return []

    def run_pipeline(self):
        from .etl import ETL
        from .data_preprocessing import DataPreprocessing

        etl = ETL()
        df_shiurim: pd.DataFrame = etl.get_shiurim_df()
        df_bookmarks: pd.DataFrame = etl.get_bookmarks_df()
        df_favorites: pd.DataFrame = etl.get_favorites_df()

        preprocessor = DataPreprocessing(df_shiurim, df_bookmarks, df_favorites)
        df_shiurim, df_bookmarks, df_favorites, df_categories = preprocessor.preprocess()
        df_user_taste = UserTaste(df_shiurim, df_bookmarks, df_categories).get_user_taste()

        df_shiurim.to_csv(f"{CleanedData.SHIURIM.value}.csv")
        df_bookmarks.to_csv(f"{CleanedData.BOOKMARKS.value}.csv")
        df_favorites.to_csv(f"{CleanedData.FAVORITES.value}.csv")
        df_categories.to_csv(f"{CleanedData.CATEGORIES.value}.csv")
        df_user_taste.to_csv(f"{CleanedData.USER_TASTE.value}.csv")
        self.__save_to_db(df_shiurim, CleanedData.SHIURIM)
        self.__save_to_db(df_bookmarks, CleanedData.BOOKMARKS)
        self.__save_to_db(df_favorites, CleanedData.FAVORITES)
        self.__save_to_db(df_categories, CleanedData.CATEGORIES)
        self.__save_to_db(df_user_taste, CleanedData.USER_TASTE)
        if self.need_to_generate_calendar():
            df_calendar: pd.DataFrame = generate_calendar()
            df_calendar.to_csv(f"{CleanedData.CALENDAR.value}.csv")
            self.__save_to_db(df_calendar, CleanedData.CALENDAR)
            logger.info("cycles_calendar generated")
    
    def need_to_generate_calendar(self) -> bool:
        cur = self.db.cursor()
        listOfTables = cur.execute(
            """SELECT name FROM sqlite_master WHERE type='table' 
            AND name='cycles_calendar'; """).fetchall()
        if listOfTables == []:
            logger.info("Calendar does not yet exist")
            return True
          

if __name__ == "__main__":
    processor = DataProcessor()
    start = time.time()
    processor.run_pipeline()
    end = time.time()
    length = round((end - start) / 60, 2)
    logger.info(f"Data Pipeline Complete: {length} min")
