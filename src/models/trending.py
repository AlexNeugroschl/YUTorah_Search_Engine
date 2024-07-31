from typing import Dict
import pandas as pd
from datetime import date, timedelta
from src.pipeline.data_processor import DataProcessor, CleanedData


class Trending:
    def __init__(self):
        dp = DataProcessor()
        self.bookmarks = dp.load_table(CleanedData.BOOKMARKS)
        self.shiurim = dp.load_table(CleanedData.SHIURIM)
        self.merged = self.__merge_shiurim(self.shiurim,self.bookmarks)

    def get_trending(self, top_n: int = 5, past_days: int = 7) -> Dict[int,str]:
        filtered = self.__filter(past_days)
        trending_shiurim = self.__get_popularity(filtered)
        return dict(zip(trending_shiurim.index[:top_n],trending_shiurim['full_details'][:top_n]))

    def get_trending_filtered(self,top_n: int = 5, past_days: int = 7, feature_key: str = None,feature_value: str = None) -> Dict[int, float]:
        filtered =  self.__filter(past_days,feature_key,feature_value)
        trending_shiurim = self.__get_popularity(filtered)
        return dict(zip(trending_shiurim.index[:top_n],trending_shiurim['full_details'][:top_n]))

    def __get_counts(self, filtered_df) -> pd.DataFrame:
        played = filtered_df[filtered_df['played'] == 1].groupby('shiur')
        queued = filtered_df.dropna(subset='queue_date').groupby('shiur')

        filtered_df['total_listens'] = played['played'].count()
        filtered_df['total_queues'] = queued['queue_date'].count()
        filtered_df = filtered_df.fillna(0)
        filtered_df['full_details'] = filtered_df['full_details'].astype(str)
        filtered_df['total_interactions'] = filtered_df['total_listens'] + \
            filtered_df['total_queues']

        return filtered_df.sort_values(by='total_interactions', ascending=False)

    def __get_popularity(self, filtered_df) -> pd.DataFrame:
        filtered_df = self.__get_counts(filtered_df)
        filtered_df['normalized_listens'] = (
            (filtered_df['total_listens'] - filtered_df['total_listens'].min()) /
            (filtered_df['total_listens'].max() - filtered_df['total_listens'].min())
        )
        filtered_df['normalized_queues'] = (
            (filtered_df['total_queues'] - filtered_df['total_queues'].min()) /
            (filtered_df['total_queues'].max() - filtered_df['total_queues'].min())
        )

        weight_listens = 0.7
        weight_queues = 0.3

        filtered_df['popularity'] = (
            (weight_listens * filtered_df['normalized_listens']) +
            (weight_queues * filtered_df['normalized_queues'])
        )

        filtered_df.drop(columns=['normalized_listens',
                     'normalized_queues'], inplace=True)

        return filtered_df.sort_values(by='popularity', ascending=False)

    def __filter(self, past_days: int = 7,feature_key: str = None, feature_value: str = None) -> pd.DataFrame:
        filtered_shiurim = self.__get_recent_interactions(past_days)
        
        if feature_key is not None and feature_value is not None:
                filtered_shiurim = filtered_shiurim[filtered_shiurim[feature_key] == feature_value]
        
        return filtered_shiurim
    
    def __get_recent_interactions(self, past_days: int = 7) -> pd.DataFrame:
        interactions = pd.DataFrame()
        today = pd.to_datetime(date.today())
        delta = today - timedelta(days=past_days)
        interactions = self.merged[(self.merged['date_played'] > delta) |
                                    (self.merged['date_downloaded'] > delta) |
                                    (self.merged['queue_date'] > delta)
                                    ]
        return interactions

    def __merge_shiurim(self, shiurim: pd.DataFrame, bookmarks: pd.DataFrame) -> pd.DataFrame:
        shiurim['name'] = shiurim['teacher_title'] + ' ' + \
            shiurim['first_name'] + ' ' + shiurim['last_name']
        shiurim.drop(columns=['teacher_title',
                     'first_name', 'last_name'], inplace=True)
        shiurim = shiurim[['shiur', 'title', 'name',
                           'category', 'middle_category', 'subcategory', 'series_name','full_details']]
        merged = pd.merge(bookmarks, shiurim, on='shiur',how='left')
        return merged