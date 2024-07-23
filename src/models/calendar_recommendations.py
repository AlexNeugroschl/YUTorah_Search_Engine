from src.pipeline.data_processor import DataProcessor, CleanedData
from enum import Enum
from datetime import date, timedelta
import pandas as pd
import re


dp = DataProcessor()
calendar = dp.load_table(CleanedData.CALENDAR)

class LearningCycle(Enum):
    DAF = ["Daf Yomi", "category_Gemara", "d_masechta", "d_num"]
    WEEKLY_DAF = ["Daf Hashvua", "category_Gemara", "dw_masechta", "dw_num"]
    MISHNAH = ["Mishna Yomi LZN Daniel Ari ben Avraham Kadesh", "category_Mishna", "m_masechta", "m_num1", "m_num2"]
    PARSHA = ["category_Parsha", "parashat"]
    NACH = ["Nach Yomi", "category_Nach", "n_sefer", "n_num"]
    YERUSHALMI = ["Yerushalmi Yomi", "category_Yerushalmi", "y_masechta", "y_num"]

def get_learning_cycle_recommendations(cycle:LearningCycle, date:date=date.today()):
     if str(date) not in calendar['date'].values:
          return None
     date_data = calendar[calendar['date'] == str(date)]
     if cycle in [LearningCycle.DAF, LearningCycle.WEEKLY_DAF, LearningCycle.NACH, LearningCycle.YERUSHALMI]:
          df = get_standard_learning(cycle, date_data)
     elif cycle == LearningCycle.PARSHA:
          df = get_parsha_recommendations(cycle, date_data)
     elif cycle == LearningCycle.MISHNAH:
          df = get_mishna_recommendation(cycle, date_data)
     else:
          return None
     # df.sort_values(by='date', inplace=True, ascending=False)
     return(df["shiur"].tolist())

def get_standard_learning(cycle:LearningCycle, row:pd.DataFrame):
     subcategory = row.iloc[0][cycle.value[2]]
     subcategory = f'[{subcategory}]' if ' ' in subcategory else subcategory
     df_categories = dp.load_table(CleanedData.CATEGORIES)
     df_shiurim = dp.load_table(CleanedData.SHIURIM)
     df_merged = pd.merge(df_categories, df_shiurim, on='shiur', suffixes=('_cat', '_shiur'))
     df = df_merged.loc[
     (df_merged[cycle.value[1]] == 1) & 
     (df_merged[row.iloc[0][cycle.value[2]]] == 1) &
     (df_merged['series_name'] == cycle.value[0])
     ].copy()
     df.loc[:, 'numbers'] = df['title'].apply(extract_numbers)
     cycle_value1 = int(row[cycle.value[3]].item() if hasattr(row[cycle.value[3]], 'item') else row[cycle.value[3]])
     filtered_df = df[df['numbers'].apply(lambda x: x[0] == cycle_value1 if len(x) > 0 else False)]
     filtered_df = filtered_df.drop(columns=['numbers'])
     return filtered_df

def get_parsha_recommendations(cycle:LearningCycle, row:pd.DataFrame):
     subcategory = row.iloc[0][cycle.value[1]]
     subcategory = f'[{subcategory}]' if ' ' in subcategory else subcategory
     df_categories = dp.load_table(CleanedData.CATEGORIES)
     df_shiurim = dp.load_table(CleanedData.SHIURIM)
     df_merged = pd.merge(df_categories, df_shiurim, on='shiur', suffixes=('_cat', '_shiur'))
     filtered_df = df_merged[
     (df_merged[cycle.value[0]] == 1) & 
     (df_merged[subcategory] == 1)
     ]
     return filtered_df

def get_mishna_recommendation(cycle:LearningCycle, row:pd.DataFrame):
     subcategory = row.iloc[0][cycle.value[2]]
     subcategory = f'[{subcategory}]' if ' ' in subcategory else subcategory
     df_categories = dp.load_table(CleanedData.CATEGORIES)
     df_shiurim = dp.load_table(CleanedData.SHIURIM)
     df_merged = pd.merge(df_categories, df_shiurim, on='shiur', suffixes=('_cat', '_shiur'))
     df = df_merged.loc[
     (df_merged[cycle.value[1]] == 1) & 
     (df_merged[row.iloc[0][cycle.value[2]]] == 1) &
     (df_merged['series_name'] == cycle.value[0])
     ].copy()
     df.loc[:, 'numbers'] = df['title'].apply(extract_numbers)
     print(df[['title', 'numbers']])
     cycle_value1 = int(row[cycle.value[3]].item() if hasattr(row[cycle.value[3]], 'item') else row[cycle.value[3]])
     cycle_value2 = int(row[cycle.value[4]].item() if hasattr(row[cycle.value[4]], 'item') else row[cycle.value[4]])
     filtered_df = df[df['numbers'].apply(lambda x: (x[0] == cycle_value1 and x[1] == cycle_value2) if len(x) > 1 else False)]
     filtered_df = filtered_df.drop(columns=['numbers'])
     return filtered_df

def extract_numbers(title):
    return [int(num) for num in re.findall(r'\b\d+\b|(?<=[:\-])\d+', title)]
