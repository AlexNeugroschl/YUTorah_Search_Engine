# TorahNavigator

## Overview

This project is a recommendation engine for YU Torah, utilizing machine learning algorithms to deliver personalized lecture suggestions to users. By analyzing user preferences and interaction history, TorahNavigator enhances the learning experience by providing tailored content that matches individual interests and study patterns.

## Project Structure

- `main.py`: Entry point of the API.
- `logging_config.py`: Basic logging configuration.
- `routers/`: Contains FastAPI routers for different recommendation endpoints.
- `models/`: Contains models for generating recommendations.
- `pipeline/`: Contains the data pipeline for handling the YUTorah DB.
- `tests/`: Contains tests for the application

## Getting Started

Follow these steps to get the project up and running:

1. **Clone the Repository**

   ```sh
   git clone https://github.com/SM24-Industrial-Software-Dev/Torah-Navigator.git
   cd Torah-Navigator
   ```

2. **Prepare the Database**

- Ensure that you have the YUTorah Database saved locally in the root directory: `yutorah_full_stats.db`

3. **Set Up Environment Variables**

- Create a `.env` file in the root directory of the project and add the path to your database:
  `DB_PATH="path/to/your/yutorah_full_stats.db"`

4. **Run the Data Pipeline**

- Run the Data Pipeline, by executing the following script: `python -m src.pipeline.data_processor`

5. **Using the DataProcessor**

- You can now import the `DataProcessor` class and use the `CleanedData` enums to load the cleaned data and test different models. Here's an example:

```
from src.pipeline.data_processor import DataProcessor, CleanedData

dp = DataProcessor()

df = dp.load_table(CleanedData.SHIURIM) # Other options: CleanedData.BOOKMARKS, CleanedData.FAVORITES, CleanedData.CATEGORIES
print(df.head())
```

## API Endpoints

#### Content Recommendations
- URL: /api/v1/content-recommendations/{user_id}
- Method: GET
- URL Parameters:
- `top_n (int)`: The number of recomendations to show. Default value is 10.
- `user_id (int)`: The ID of the user to get content recommendations for.
- Response: Dict[int, str] - A dictionary with Shiur IDs and their corresponding titles, speakers, and categories.
#### Becuase You Liked Recommendations
- URL: /api/v1/because-you-listened-recommendations/{user_id}
- Method: GET
- URL Parameters:
- `top_n (int)`: The number of recomendations to show. default value is 5.
- `user_id (int)`: The ID of the user to get 'because-you-liked' recommendations for.
- Response: Dict[int, str] - A dictionary with Shiur IDs and their corresponding titles, speakers, and categories.
#### Trending Recommendations
- URL: /api/v1/trending
- Method: GET
- URL Parameters:
- `top_n (int)`: The number of recomendations to show. Default value is 5.
- `past_days (int)`: The number of past days to consider for trending items. Default value is 7.
- Response: Dict[int, str] - A dictionary with Shiur IDs and their corresponding titles, speakers, and categories.
#### Trending Filtered Recommendations
- URL: /api/v1/trending/filtered/{feature_key}={feature_value}
- Method: GET
- URL Parameters:
- `top_n (int)`: The number of recomendations to show. Default value is 5.
- `past_days (int)`: The number of past days to consider for trending items. Default value is 7.
- `feature_key (str)`: The feauture key to filter trending shiurim by. Valid feature keys include:
  - `name` (of speaker)
  - `category`
  - `middle_category`
  - `subcategory`
  - `series_name`
- `feature_value (str)`: The feature value to filter trending shiurim by. The format for name is speaker title, first name, and last name, all capitalized with a space between words. All category filters should be capitalized with a space between multiple words. Categories include:
  - `category`: Broader categories such as Parsha, Gemara, Machshava, Halacha, among others.
  - `middle_category`: Holidays, books of Chumash, Seder of Mishna/Gemara, Chelek in Shulchan Aruch, among others.
  - `subcategory`: Names of the parsha, mesechtot, holiday names, section of Halacha, among others.
- Response: Dict[int, str] - A dictionary with Shiur IDs and their corresponding titles, speakers, and categories.

## Coding Standards

- Follow PEP 8 for Python code style.
- Write meaningful commit messages and justifications for using certain techniques when appropriate.
