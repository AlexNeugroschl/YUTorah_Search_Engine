from typing import Dict, List
from .base import BaseModel
import pandas as pd
from datetime import date, timedelta
from src.pipeline.data_processor import DataProcessor, CleanedData


class Trending:
    def __init__(self):
        dp = DataProcessor()
        self.bookmarks = dp.load_table(CleanedData.BOOKMARKS)
        self.shiurim = dp.load_table(CleanedData.SHIURIM)
        self.merged = self.__merge_shiurim(self.shiurim,self.bookmarks)

    def get_trending(self, top_n: int = 5, past: int = 7) -> Dict[int,str]:
        filtered = self.__filter(past)
        trending_shiurim = self.__get_popularity(filtered)
        return dict(zip(trending_shiurim.index[:top_n],trending_shiurim['full_details'][:top_n]))

    def get_trending_filtered(self,top_n: int = 5, past: int = 7, key: str = None,value: str = None) -> Dict[int, float]:
        filtered =  self.__filter(past,key,value)
        trending_shiurim = self.__get_popularity(filtered)
        return dict(zip(trending_shiurim.index[:top_n],trending_shiurim['full_details'][:top_n]))

    def __get_top_recent_shiurim(self, merged: pd.DataFrame, past: int = 7) -> pd.DataFrame:
        shiurim = merged.copy()
        interactions = self.__get_recent_interactions(past)

        played = interactions[interactions['played'] == 1].groupby('shiur')
        queued = interactions.dropna(subset='queue_date').groupby('shiur')

        shiurim['total_listens'] = played['played'].count()
        shiurim['total_queues'] = queued['queue_date'].count()
        shiurim = shiurim.fillna(0)
        shiurim['full_details'] = shiurim['full_details'].astype(str)
        shiurim['total_interactions'] = shiurim['total_listens'] + \
            shiurim['total_queues']

        return shiurim.sort_values(by='total_interactions', ascending=False)

    def __get_popularity(self,merged : pd.DataFrame) -> pd.DataFrame:
        shiurim = self.__get_top_recent_shiurim(merged)
        shiurim['normalized_listens'] = (
            (shiurim['total_listens'] - shiurim['total_listens'].min()) /
            (shiurim['total_listens'].max() - shiurim['total_listens'].min())
        )
        shiurim['normalized_queues'] = (
            (shiurim['total_queues'] - shiurim['total_queues'].min()) /
            (shiurim['total_queues'].max() - shiurim['total_queues'].min())
        )

        weight_listens = 0.7
        weight_queues = 0.3

        shiurim['popularity'] = (
            (weight_listens * shiurim['normalized_listens']) +
            (weight_queues * shiurim['normalized_queues'])
        )

        shiurim.drop(columns=['normalized_listens',
                     'normalized_queues'], inplace=True)

        return shiurim.sort_values(by='popularity', ascending=False)

    def __filter(self, past: int = 7,key: str = None, value: str = None) -> pd.DataFrame:
        filtered_shiurim = self.__get_recent_interactions(past)
        
        if key is not None and value is not None:
                filtered_shiurim = filtered_shiurim[filtered_shiurim[key] == value]
        
        return filtered_shiurim
    
    def __get_recent_interactions(self, past: int = 7) -> pd.DataFrame:
        interactions = self.merged.copy()
        today = pd.to_datetime(date.today())
        delta = today - timedelta(days=past)
        interactions = interactions[(interactions['date_played'] > delta) |
                                    (interactions['date_downloaded'] > delta) |
                                    (interactions['queue_date'] > delta)
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