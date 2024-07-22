import pandas as pd
from .db_connection import db_connection
from ..logging_config import setup_logging
from ..decorators import log_ram_usage, log_and_time_execution

logger = setup_logging()


class ETL:
    def __init__(self, chunk_size: int = 100_000):
        self.conn = db_connection()
        self.chunk_size = chunk_size
        logger.info("ETL instance created")

    @log_and_time_execution
    def get_favorites_df(self) -> pd.DataFrame:
        query_fav = """
        SELECT 
            ufUserKey as 'user',
            ufForeignKey as 'key',
            ufType as 'favorite_type',
            ufDateAdded as 'date_favorite_added'
        FROM userFavorites
        ORDER BY date_favorite_added DESC
        """
        fav_chunks = pd.read_sql_query(
            query_fav, self.conn, chunksize=self.chunk_size, parse_dates=['date_favorite_added'])
        df_fav = pd.concat(fav_chunks)
        df_fav = df_fav.sort_values(by='user', ascending=True)
        return df_fav

    @log_and_time_execution
    def get_bookmarks_df(self) -> pd.DataFrame:
        query_usb = """
    SELECT
        usbUserKey as 'user',
        usbShiurKey as 'shiur',
        usbSessionID as 'session',
        usbBookmarkType as 'bookmark',
        usbBookmarkTimeStamp as 'timestamp',
        usbDateAddedToQueue as 'queue_date',
        usbIsPlayed as 'played',
        usbDatePlayed as 'date_played',
        usbIsDownloaded as 'downloaded',
        usbDateDownloaded as 'date_downloaded',
        s.shiurMediaLength as 'duration'
    FROM userShiurBookmarks usb
    JOIN shiurim s ON s.shiurID = usb.usbShiurKey 
    WHERE usbUserKey IS NOT NULL
        AND usbBookmarkType IN ('history','isPlayed','lastPlayed','queue')
    ORDER BY shiur DESC
    """
        usb_chunks = pd.read_sql_query(
            query_usb, self.conn, chunksize=self.chunk_size, parse_dates=['date', 'queue_date', 'date_played', 'date_downloaded'])
        df_usb = pd.concat(usb_chunks)
        df_usb = df_usb.sort_values(by='user', ascending=True)
        return df_usb

    @log_and_time_execution
    def get_shiurim_df(self) -> pd.DataFrame:
        # Merge with categories
        df_shiurim = pd.merge(self.__get_shiurim_teachers(),
                              self.__get_cat(), on='shiur')

        # Merge with locations
        df_shiurim = pd.merge(
            df_shiurim, self.__get_locations(), on='loc_id', how='left')

        # Merge with series
        df_shiurim = pd.merge(
            df_shiurim, self.__get_series(), on='series_id', how='left')

        # Drop unnecessary columns
        df_shiurim = df_shiurim.drop(columns=['loc_id', 'series_id'])
        df_shiurim.sort_values("shiur", ascending=False)
        return df_shiurim

    def __get_shiurim_teachers(self) -> pd.DataFrame:
        # Query for shiurim and teachers
        query_shiurim = """
        SELECT 
            s.shiurID AS shiur, 
            s.shiurTitle AS title, 
            t.teacherTitle AS teacher_title, 
            t.teacherLastName AS last_name, 
            t.teacherFirstName AS first_name, 
            s.shiurDate AS date, 
            s.shiurLanguage AS language, 
            s.shiurMediaLength AS duration, 
            s.shiurKeywords AS keywords, 
            s.shiurLocationKey AS loc_id, 
            s.shiurSeriesKey AS series_id
        FROM 
            shiurim s 
        INNER JOIN 
            shiurTeachers st ON s.shiurID = st.shiurTeacherShiurKey 
        INNER JOIN 
            teachers t ON st.shiurTeacherTeacherKey = t.teacherID 
        WHERE 
            t.teacherIsHidden = 0 AND s.shiurIsVisibleOnYuTorah = 1
        """
        return pd.read_sql_query(query_shiurim, self.conn, parse_dates=['date'])

    def __get_cat(self) -> pd.DataFrame:
        # Query for categories and subcategories
        query_cat = """
        SELECT 
            shiurCategoryShiurKey AS shiur, 
            c.categoryShortName AS category, 
            s.subcategoryMiddleTier AS middle_category, 
            s.subcategoryName AS subcategory
        FROM 
            shiurCategories sc 
        INNER JOIN 
            subcategories s ON sc.shiurCategorySubcategoryKey = s.subcategoryID 
        INNER JOIN 
            categories c ON s.subcategoryCategoryKey = c.categoryID 
        """
        return pd.read_sql_query(query_cat, self.conn)

    def __get_locations(self) -> pd.DataFrame:
        # Query for locations
        query_loc = """
        SELECT 
            locationID AS loc_id, 
            locationName AS location, 
            locationMiddleTier AS location_type
        FROM 
            locations
        """
        return pd.read_sql_query(query_loc, self.conn)

    def __get_series(self) -> pd.DataFrame:
        # Query for series
        query_series = """
        SELECT 
            seriesID AS series_id, 
            seriesName AS series_name, 
            seriesDescription AS series_description
        FROM 
            series
        """
        return pd.read_sql_query(query_series, self.conn)
