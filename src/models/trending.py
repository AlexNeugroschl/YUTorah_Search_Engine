from typing import Dict, List
from .base import BaseModel
import pandas as pd
from datetime import date, timedelta
from src.pipeline.data_processor import DataProcessor, CleanedData


class Trending(BaseModel):
    def __init__(self):
        super().__init__()
        dp = DataProcessor()
        self.bookmarks = dp.load_table(CleanedData.BOOKMARKS)
        self.shiurim = dp.load_table(CleanedData.SHIURIM)
        self.merged = self.__merge_shiurim(self.shiurim,self.bookmarks)

    def get_recommendations(self, user_id: str = None, top_n: int = 5, past: int = 7, *args, **kwargs) -> List[int]:
        for key,value in kwargs.items():
            if key in self.shiurim.columns:
                trending_filtered =  self.__get_filtered(key=value)
                return self.__get_popularity(trending_filtered,past).index[:top_n].tolist()
        trending_shiurim = self.__get_popularity(
            self.__get_top_recent_shiurim(self.bookmarks, past))
        return trending_shiurim.index[:top_n].tolist()

    def get_weighted_recommendations(self, user_id: str = None, top_n: int = 5, past: int = 7, *args, **kwargs) -> Dict[int, float]:
        trending_shiurim = self.__get_popularity(
            self.__get_top_recent_shiurim(self.bookmarks, past))
        return dict(zip(trending_shiurim.index[:top_n], trending_shiurim['popularity'][:top_n]))

    def __get_recent_interactions(self, bookmarks: pd.DataFrame, past: int = 7) -> pd.DataFrame:
        interactions = bookmarks
        today = pd.to_datetime(date.today())
        delta = today - timedelta(days=past)
        interactions = interactions[(interactions['date_played'] > delta) |
                                    (interactions['date_downloaded'] > delta) |
                                    (interactions['queue_date'] > delta)
                                    ]
        return interactions

    def __get_top_recent_shiurim(self, bookmarks: pd.DataFrame, past: int = 7) -> pd.DataFrame:
        shiurim = pd.DataFrame()
        interactions = self.__get_recent_interactions(bookmarks, past)

        played = interactions[interactions['played'] == 1].groupby('shiur')
        queued = interactions.dropna(subset='queue_date').groupby('shiur')

        shiurim['total_listens'] = played['played'].count()
        shiurim['total_queues'] = queued['queue_date'].count()
        shiurim = shiurim.fillna(0)
        shiurim['total_interactions'] = shiurim['total_listens'] + \
            shiurim['total_queues']

        return shiurim.sort_values(by='total_interactions', ascending=False)

    def __get_popularity(self, shiurim: pd.DataFrame) -> pd.DataFrame:
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

    def __merge_shiurim(self, shiurim: pd.DataFrame, bookmarks: pd.DataFrame) -> pd.DataFrame:
        shiurim['name'] = shiurim['teacher_title'] + ' ' + \
            shiurim['first_name'] + ' ' + shiurim['last_name']
        shiurim.drop(columns=['teacher_title',
                     'first_name', 'last_name'], inplace=True)
        shiurim = shiurim[['shiur', 'title', 'name', 'duration',
                           'category', 'middle_category', 'subcategory', 'series_name']]
        merged = pd.merge(bookmarks, shiurim, on='shiur')
        return merged

    def __get_filtered(self, **kwargs) -> pd.DataFrame:
        trending_shiurim = self.__get_top_recent_shiurim(self.merged)
        for key, value in kwargs.items():
            trending_shiurim = trending_shiurim[trending_shiurim[key] == value]
        return trending_shiurim
