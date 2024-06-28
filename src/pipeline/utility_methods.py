import numpy as np
import pandas as pd
from typing import Union, List, Dict

# TODO: Currently doesn't work, don't want to change until all data processing is done


class UtilityMethods:
    def __init__(self, dl: DataLayer):
        self.dl = dl

    def get_shiurs_listened_by_user(self, user_id: int) -> np.ndarray:
        interactions_df = self.dl.get_user_interactions(user_id)
        listened_list = interactions_df[interactions_df['bookmark'].isin(
            ['lastPlayed', 'isPlayed'])]
        return listened_list['shiur'].unique().astype(int)

    def get_shiurs_queued_by_user(self, user_id: int) -> np.ndarray:
        interactions_df = self.dl.get_user_interactions(user_id)
        queued_list = interactions_df[interactions_df['bookmark'].isin(
            ['queue', 'history'])]
        return queued_list['shiur'].unique().astype(int)

    def get_user_favorite_teachers(self, user_id: int) -> np.ndarray:
        favorites_df = self.dl.get_user_favorites(user_id)
        return favorites_df[favorites_df['favorite_type'] == 'teacher']['key'].unique().astype(int)

    def get_user_favorite_series(self, user_id: int) -> np.ndarray:
        favorites_df = self.dl.get_user_favorites(user_id)
        return favorites_df[favorites_df['favorite_type'] == 'series']['key'].unique().astype(int)

    def get_shiur_listen_history(self, shiur_id: int) -> pd.DataFrame:
        interactions_df = self.dl.get_shiur_interactions(shiur_id)
        listened_list = interactions_df[(interactions_df['user'].notna()) &
                                        (interactions_df['bookmark'].isin(['lastPlayed', 'isPlayed']))]
        return listened_list

    def get_users_listened_to_shiur(self, shiur_id: int) -> np.ndarray:
        listened_list = self.get_shiur_listen_history(shiur_id)
        return listened_list['user'].unique().astype(int)

    def get_shiur_details(self, shiur_id: int) -> pd.DataFrame:
        return self.dl.get_shiur_details(shiur_id)

    def get_shiurs_by_teacher(self, teacher_id: int) -> np.ndarray:
        shiurs_df = self.dl.get_shiurs_by_teacher(teacher_id)
        return shiurs_df['shiur'].unique().astype(int)

    def get_categories_by_shiur(self, shiur_ids: Union[int, List[int]]) -> Dict[int, np.ndarray]:
        if isinstance(shiur_ids, int):
            shiur_ids = [shiur_ids]

        shiur_cat_dict = {}
        for id in shiur_ids:
            shiur_cat_dict[id] = self.get_category_by_shiur(id)

        return shiur_cat_dict

    def get_category_by_shiur(self, shiur_id: int) -> np.ndarray:
        shiurs_df = self.dl.get_shiur_details(shiur_id)
        return shiurs_df['category'].unique()
