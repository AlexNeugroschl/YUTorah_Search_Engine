from src.pipeline.data_processor import DataProcessor, CleanedData
from enum import Enum
from datetime import date, timedelta, datetime
import pandas as pd
import re


class LearningCycle(Enum):
    DAF = ["Daf Yomi", "category_Gemara", "d_masechta", "d_num"]
    WEEKLY_DAF = ["Daf Hashvua", "category_Gemara", "dw_masechta", "dw_num"]
    MISHNAH = ["Mishna Yomi LZN Daniel Ari ben Avraham Kadesh", "category_Mishna", "m_masechta", "m_num1", "m_num2"]
    PARSHA = ["category_Parsha", "parashat"]
    NACH = ["Nach Yomi", "category_Nach", "n_sefer", "n_num"]
    YERUSHALMI = ["Yerushalmi Yomi", "category_Yerushalmi", "y_masechta", "y_num"]


class CycleRecommendations():
     def __init__(self):
          self.dp = DataProcessor()
          self.calendar = self.dp.load_table(CleanedData.CALENDAR)
          df_categories = self.dp.load_table(CleanedData.CATEGORIES)
          df_shiurim = self.dp.load_table(CleanedData.SHIURIM)
          self.df_merged = pd.merge(df_categories, df_shiurim, on='shiur', suffixes=('_cat', '_shiur'))

     def get_all_recommendations(self, date:date=date.today()):
          if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
          all_recommendations = []
          for cycle in LearningCycle:
               recommendations = self.get_learning_cycle_recommendations(cycle, date)
               all_recommendations.extend(recommendations)
          all_recommendations.extend(self.get_holiday_recommendations(date, date+timedelta(days=3)))
          return all_recommendations

     def get_daf_recommendations(self, date:date=date.today):
          return self.get_learning_cycle_recommendations(LearningCycle.DAF, date)
     
     def get_weekly_daf_recommendations(self, date:date=date.today):
          return self.get_learning_cycle_recommendations(LearningCycle.DAF, date)
     
     def get_mishnah_recommendations(self, date:date=date.today):
          return self.get_learning_cycle_recommendations(LearningCycle.WEEKLY_DAF, date)
     
     def get_parsha_recommendations(self, date:date=date.today):
          return self.get_learning_cycle_recommendations(LearningCycle.PARSHA, date)
     
     def get_nach_recommendations(self, date:date=date.today):
          return self.get_learning_cycle_recommendations(LearningCycle.NACH, date)
     
     def get_yerushalmi_recommendations(self, date:date=date.today):
          return self.get_learning_cycle_recommendations(LearningCycle.YERUSHALMI, date)
     
     def get_learning_cycle_recommendations(self, cycle:LearningCycle, date:date=date.today()):
          if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()
          if str(date) not in self.calendar['date'].values:
               return []
          date_data = self.calendar[self.calendar['date'] == str(date)]
          if cycle in [LearningCycle.DAF, LearningCycle.WEEKLY_DAF, LearningCycle.NACH, LearningCycle.YERUSHALMI]:
               df = self.__calculate_standard_learning(cycle, date_data)
          elif cycle == LearningCycle.PARSHA:
               df = self.__calculate_parsha_recommendations(cycle, date_data)
          elif cycle == LearningCycle.MISHNAH:
               df = self.__calculate_mishna_recommendation(cycle, date_data)
          else:
               return []
          return(df["shiur"].tolist())

     def __calculate_standard_learning(self, cycle:LearningCycle, row:pd.DataFrame):
          subcategory = row.iloc[0][cycle.value[2]]
          subcategory = f'[{subcategory}]' if ' ' in subcategory else subcategory
          df = self.df_merged.loc[
          (self.df_merged[cycle.value[1]] == 1) & 
          (self.df_merged[row.iloc[0][cycle.value[2]]] == 1) &
          (self.df_merged['series_name'] == cycle.value[0])
          ].copy()
          df.loc[:, 'numbers'] = df['title'].apply(self.__extract_numbers)
          cycle_value1 = int(row[cycle.value[3]].item() if hasattr(row[cycle.value[3]], 'item') else row[cycle.value[3]])
          filtered_df = df[df['numbers'].apply(lambda x: x[0] == cycle_value1 if len(x) > 0 else False)]
          filtered_df = filtered_df.drop(columns=['numbers'])
          return filtered_df

     def __calculate_parsha_recommendations(self, cycle:LearningCycle, row:pd.DataFrame):
          subcategory = row.iloc[0][cycle.value[1]]
          subcategory = f'[{subcategory}]' if ' ' in subcategory else subcategory
          filtered_df = self.df_merged[
          (self.df_merged[cycle.value[0]] == 1) & 
          (self.df_merged[subcategory] == 1)
          ]
          return filtered_df

     def __calculate_mishna_recommendation(self, cycle:LearningCycle, row:pd.DataFrame):
          subcategory = row.iloc[0][cycle.value[2]]
          subcategory = f'[{subcategory}]' if ' ' in subcategory else subcategory
          df = self.df_merged.loc[
          (self.df_merged[cycle.value[1]] == 1) & 
          (self.df_merged[row.iloc[0][cycle.value[2]]] == 1) &
          (self.df_merged['series_name'] == cycle.value[0])
          ].copy()
          df.loc[:, 'numbers'] = df['title'].apply(self.__extract_numbers)
          num1 = int(row[cycle.value[3]].item() if hasattr(row[cycle.value[3]], 'item') else row[cycle.value[3]])
          num2 = int(row[cycle.value[4]].item() if hasattr(row[cycle.value[4]], 'item') else row[cycle.value[4]])
          filtered_df = df[df['numbers'].apply(lambda x: (x[0] == num1 and x[1] == num2) if len(x) > 1 else False)]
          filtered_df = filtered_df.drop(columns=['numbers'])
          return filtered_df

     def __extract_numbers(self, title):
          return [int(num) for num in re.findall(r'\b\d+\b|(?<=[:\-])\d+', title)]

     def get_holiday_recommendations(self, start_date:date=date.today(), end_date:date=date.today()+timedelta(days=3)):
          if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
          if str(start_date) not in self.calendar['date'].values:
               return []
          holiday_data = self.calendar[(self.calendar['date'] >= str(start_date)) & (self.calendar['date'] <= str(end_date))]
          no_holiday = holiday_data['holiday'].isna().all()
          no_roshchodesh = pd.isna(holiday_data['roshchodesh'].iloc[0])
          if not no_holiday:
               first_holiday = holiday_data['holiday'].dropna().iloc[0]
               filtered_df = self.df_merged[(self.df_merged[first_holiday] == 1) & (self.df_merged['category_Holidays'] == 1)]
               return(filtered_df["shiur"].tolist())
          elif not no_roshchodesh:
               first_roshchodesh = holiday_data['holiday'].dropna().iloc[0]
               filtered_df = self.df_merged[(self.df_merged[first_roshchodesh] == 1) & (self.df_merged['category_Holidays'] == 1)]
               return(filtered_df["shiur"].tolist())
          else:
               return []
