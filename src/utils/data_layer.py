import sqlite3
import pandas as pd
from contextlib import closing


# Queries use paramaters over f-string to protect against SQL injection attacks and optimized for performance.
class DataLayer:
    def __init__(self, db_path):
        self.db_path = db_path

    def execute_query(self, query, params=None):
        with closing(sqlite3.connect(self.db_path)) as conn:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            else:
                return pd.read_sql_query(query, conn)

    def get_user_interactions(self, user_id: int):
        query = "SELECT * FROM userShiurBookmarks WHERE usbUserKey = ?"
        return self.execute_query(query, params=(user_id,))

    def get_user_favorites(self, user_id: int):
        query = "SELECT * FROM userFavorites WHERE ufUserKey = ?"
        return self.execute_query(query, params=(user_id,))

    def get_shiur_interactions(self, shiur_id: int):
        query = "SELECT * FROM userShiurBookmarks WHERE usbShiurKey = ?"
        return self.execute_query(query, params=(shiur_id,))

    def get_shiurs_by_teacher(self, teacher_id: int):
        query = """
        SELECT *
        FROM teachers t
        JOIN shiurTeachers st ON t.teacherID = st.shiurTeacherTeacherKey 
        JOIN shiurim s ON st.shiurTeacherShiurKey = s.shiurID 
        WHERE teacherID = ?
        """
        return self.execute_query(query, params=(teacher_id,))

    def get_shiur_details(self, shiur_id: int):
        query = """
        SELECT *
        FROM teachers t 
        JOIN shiurTeachers st ON t.teacherID = st.shiurTeacherTeacherKey 
        JOIN shiurim s ON st.shiurTeacherShiurKey = s.shiurID 
        LEFT JOIN locations l ON l.locationID = s.shiurLocationKey 
        LEFT JOIN shiurCategories sc ON sc.shiurCategoryShiurKey = s.shiurID 
        LEFT JOIN subcategories s2 ON sc.shiurCategorySubcategoryKey = s2.subcategoryID 
        LEFT JOIN categories c ON s2.subcategoryCategoryKey = c.categoryID 
        LEFT JOIN series s3 ON s.shiurSeriesKey = s3.seriesID 
        WHERE s.shiurID = ?
        """
        return self.execute_query(query, params=(shiur_id,))

    def get_all_interactions(self):
        query = "SELECT * FROM userShiurBookmarks"
        return self.execute_query(query)

    def get_all_shiurim(self):
        query = "SELECT * FROM shiurim"
        return self.execute_query(query)
