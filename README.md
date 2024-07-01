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

## Coding Standards

- Follow PEP 8 for Python code style.
- Write meaningful commit messages and justifications for using certain techniques when appropriate.
