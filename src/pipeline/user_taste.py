import json
import pandas as pd
import numpy as np
from scipy.stats import entropy
from ..decorators import log_and_time_execution


class UserTaste:
    def __init__(self, df_shiurim: pd.DataFrame, df_bookmarks: pd.DataFrame, df_categories: pd.DataFrame):
        self.df_shiurim = df_shiurim
        self.df_bookmarks = df_bookmarks
        self.df_categories = df_categories
        self.shiur_stats_df = pd.DataFrame(index=df_bookmarks['user'].unique())

        self.listened_df = df_bookmarks[(df_bookmarks['played'] == 1) & (df_bookmarks['bookmark'] == 'lastPlayed')]
        self.queued_df = df_bookmarks[~df_bookmarks['queue_date'].isna()]
        self.downloaded_df = df_bookmarks[df_bookmarks['downloaded'] == 1]

        self.df_bookmark_shiur_df = pd.merge(df_bookmarks, df_shiurim, on='shiur')
        self.df_bookmark_shiur_df['name'] = self.df_bookmark_shiur_df['first_name'] + \
            ' ' + self.df_bookmark_shiur_df['last_name']

        self.user_categories_df = self.df_bookmarks[['user', 'shiur']].merge(self.df_categories, on='shiur')

    @log_and_time_execution
    def get_user_taste(self) -> pd.DataFrame:
        # Calculate different metrics
        self.__get_listens()
        self.__get_queues()
        self.__get_downloads()
        self.__get_top_teacher()
        self.__get_top_categories()
        self.__get_top_series()
        self.__get_category_diversity()
        self.__get_average_listen_of_shiur()
        self.__get_average_listen_percentage()
        self.__get_most_recent_shiurs()

        # Serialize list columns to JSON strings
        list_columns = ['top_teachers', 'top_categories', 'top_middle_categories',
                        'top_subcategories', 'most_recent_shiurs_id', 'most_recent_shiur_details']
        for col in list_columns:
            if col in self.shiur_stats_df.columns:
                self.shiur_stats_df[col] = self.shiur_stats_df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else x)

        return self.shiur_stats_df

    def __get_listens(self):
        listen_counts = self.listened_df.groupby('user')['played'].count()
        listen_counts = listen_counts.reindex(self.shiur_stats_df.index).fillna(0).astype('int16')
        self.shiur_stats_df['total_listens'] = listen_counts

    def __get_queues(self):
        queue_counts = self.queued_df.groupby('user')['queue_date'].count()
        queue_counts = queue_counts.reindex(self.shiur_stats_df.index).fillna(0).astype('int16')
        self.shiur_stats_df['total_queues'] = queue_counts

    def __get_downloads(self):
        download_counts = self.downloaded_df.groupby('user')['downloaded'].count()
        download_counts = download_counts.reindex(self.shiur_stats_df.index).fillna(0).astype('int16')
        self.shiur_stats_df['total_downloads'] = download_counts

    def __get_top_teacher(self):
        top_teachers = self.df_bookmark_shiur_df.groupby('user')['name'].apply(
            lambda x: x.value_counts().nlargest(3).index.tolist())
        self.shiur_stats_df['top_teachers'] = top_teachers.reindex(self.shiur_stats_df.index)

    def __get_top_categories(self):
        category_columns = [col for col in self.df_categories.columns if col.startswith('category')]
        middle_category_columns = [col for col in self.df_categories.columns if col.startswith('middle_category')]
        subcategory_columns = [col for col in self.df_categories.columns if col.startswith('subcategory')]

        def get_top_n_categories(df, columns, n=3):
            category_counts = df.groupby('user')[columns].sum()
            return category_counts.apply(lambda x: x.nlargest(n).index.tolist(), axis=1)

        top_categories = get_top_n_categories(self.user_categories_df, category_columns, 3)
        top_middle_categories = get_top_n_categories(self.user_categories_df, middle_category_columns, 3)
        top_subcategories = get_top_n_categories(self.user_categories_df, subcategory_columns, 3)

        self.shiur_stats_df['top_categories'] = top_categories.reindex(self.shiur_stats_df.index)
        self.shiur_stats_df['top_middle_categories'] = top_middle_categories.reindex(self.shiur_stats_df.index)
        self.shiur_stats_df['top_subcategories'] = top_subcategories.reindex(self.shiur_stats_df.index)

    def __get_top_series(self):
        top_series = self.df_bookmark_shiur_df.groupby('user')['series_name'].apply(lambda x: x.value_counts().idxmax())
        self.shiur_stats_df['favorite_series'] = top_series.reindex(self.shiur_stats_df.index).fillna("")

    def __get_category_diversity(self):
        user_categories = self.user_categories_df.copy()
        user_categories.drop(columns=['shiur'], inplace=True)
        category_preferences = user_categories.groupby('user').sum()

        def calculate_entropy(row):
            row_non_zero = row[row > 0]
            return entropy(row_non_zero)

        category_preferences = category_preferences.div(category_preferences.sum(axis=1), axis=0)
        diversity_df = category_preferences.apply(calculate_entropy, axis=1).reset_index()
        diversity_df.columns = ['user', 'category_diversity']

        mode_value = diversity_df['category_diversity'].mode()[0]
        diversity_df['category_diversity'] = diversity_df['category_diversity'].fillna(mode_value)

        max_entropy = np.log(len(self.df_categories.columns))
        diversity_df['category_diversity'] = (diversity_df['category_diversity'] / max_entropy).round(2)

        diversity_df = diversity_df.set_index('user').reindex(self.shiur_stats_df.index)
        self.shiur_stats_df['category_diversity'] = diversity_df['category_diversity']

    def __get_average_listen_of_shiur(self):
        avg_listen_duration = self.listened_df.groupby('user')['duration'].mean()
        avg_listen_duration = avg_listen_duration.reindex(self.shiur_stats_df.index)
        self.shiur_stats_df['average_duration_of_shiur'] = avg_listen_duration.round(2)

        avg_queue_duration = self.queued_df.groupby('user')['duration'].mean()
        avg_queue_duration = avg_queue_duration.reindex(self.shiur_stats_df.index)
        self.shiur_stats_df['average_duration_of_shiur'] = self.shiur_stats_df['average_duration_of_shiur'].fillna(
            avg_queue_duration.round(2))

        overall_mean_listen_duration = avg_listen_duration.mean().round(2)
        self.shiur_stats_df['average_duration_of_shiur'] = self.shiur_stats_df['average_duration_of_shiur'].fillna(
            overall_mean_listen_duration)

    def __get_average_listen_percentage(self):
        avg_listen_percentage = self.listened_df.groupby('user')['listen_percentage'].mean()
        avg_listen_percentage = avg_listen_percentage.reindex(self.shiur_stats_df.index)
        self.shiur_stats_df['average_listen_percentage'] = avg_listen_percentage.round(2)

        avg_queue_percentage = self.queued_df.groupby('user')['listen_percentage'].mean()
        avg_queue_percentage = avg_queue_percentage.reindex(self.shiur_stats_df.index)
        self.shiur_stats_df['average_listen_percentage'] = self.shiur_stats_df['average_listen_percentage'].fillna(
            avg_queue_percentage.round(2))

        overall_mean_listen_percentage = avg_listen_percentage.mean().round(2)
        self.shiur_stats_df['average_listen_percentage'] = self.shiur_stats_df['average_listen_percentage'].fillna(
            overall_mean_listen_percentage)
        self.shiur_stats_df['average_listen_percentage'] = self.shiur_stats_df['average_listen_percentage'].replace(
            0, overall_mean_listen_percentage)

    def __get_most_recent_shiurs(self):
        listened_sorted_df = self.listened_df.sort_values(by=['user', 'date_played'], ascending=False)
        most_recent_shiurs = listened_sorted_df.groupby('user')['shiur'].agg(lambda x: list(x)[:3])

        queued_sorted_df = self.queued_df.sort_values(by=['user', 'date_played'], ascending=False)
        queued_shiurs = queued_sorted_df.groupby('user')['shiur'].agg(lambda x: list(x)[:3])

        def get_combined_recent_shiurs(row):
            listened = row['listened'] if isinstance(row['listened'], list) else []
            queued = row['queued'] if isinstance(row['queued'], list) else []
            if len(listened) < 3:
                combined = listened + queued[:3 - len(listened)]
                return combined
            return listened

        combined_recent_shiurs = pd.DataFrame({
            'listened': most_recent_shiurs,
            'queued': queued_shiurs
        }).apply(lambda row: get_combined_recent_shiurs(row), axis=1)

        self.shiur_stats_df['most_recent_shiurs_id'] = combined_recent_shiurs.reindex(
            self.shiur_stats_df.index).fillna({})

        # Get the full_details for each shiur ID
        shiur_details_map = self.df_shiurim.set_index('shiur')['full_details'].to_dict()

        def get_shiur_details(shiur_ids):
            if isinstance(shiur_ids, list):
                return [shiur_details_map.get(shiur_id, '') for shiur_id in shiur_ids]
            return []

        self.shiur_stats_df['most_recent_shiur_details'] = self.shiur_stats_df['most_recent_shiurs_id'].apply(
            get_shiur_details)
