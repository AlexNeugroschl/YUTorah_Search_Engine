from src.utils.data_layer import DataLayer
import numpy as np
import pandas as pd


class UtilityMethods:
    def __init__(self, dl: DataLayer):
        self.dl = dl

    def get_shiurs_listened_by_user(self, user_id: int) -> np.ndarray:
        interactions_df = self.dl.get_user_interactions(user_id)
        listened_list = interactions_df[interactions_df['usbBookmarkType'].isin(
            ['lastPlayed', 'isPlayed'])]
        return listened_list['usbShiurKey'].unique().astype(int)

    def get_shiurs_queued_by_user(self, user_id: int) -> np.ndarray:
        interactions_df = self.dl.get_user_interactions(user_id)
        queued_list = interactions_df[interactions_df['usbBookmarkType'].isin(
            ['queue', 'history'])]
        return queued_list['usbShiurKey'].unique().astype(int)

    def get_shiur_listen_history(self, shiur_id: int) -> pd.DataFrame:
        interactions_df = self.dl.get_shiur_interactions(shiur_id)
        listened_list = interactions_df[(interactions_df['usbUserKey'].notna()) &
                                        (interactions_df['usbBookmarkType'].isin(['lastPlayed', 'isPlayed']))]
        return listened_list

    def get_users_listened_to_shiur(self, shiur_id: int) -> np.ndarray:
        listened_list = self.get_shiur_listen_history(shiur_id)
        return listened_list['usbUserKey'].unique().astype(int)

    def get_user_favorite_teachers(self, user_id: int) -> np.ndarray:
        favorites_df = self.dl.get_user_favorites(user_id)
        return favorites_df[favorites_df['ufType'] == 'teacher']['ufForeignKey'].unique().astype(int)

    def get_user_favorite_series(self, user_id: int) -> np.ndarray:
        favorites_df = self.dl.get_user_favorites(user_id)
        return favorites_df[favorites_df['ufType'] == 'series']['ufForeignKey'].unique().astype(int)

    def get_shiur_details(self, shiur_id: int) -> pd.DataFrame:
        return self.dl.get_shiur_details(shiur_id)
