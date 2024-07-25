from fastapi import APIRouter, HTTPException
from typing import Dict
from ..logging_config import setup_logging
from src.models.content_filtering_v1 import ContentFiltering
from src.models.calendar_recommendations import CycleRecommendations, LearningCycle
from datetime import date, timedelta

router = APIRouter()
content_filtering = ContentFiltering()
cycle_recommendations = CycleRecommendations()

logger = setup_logging()


@router.get("/content-recommendations/{shiur_id}", response_model=Dict[int, str])
def get_recommendations(shiur_id: int, top_n: int = 5):
    try:
        recommendations = content_filtering.get_recommendations(
            shiur_id, top_n)
        return recommendations
    except KeyError:
        logger.error(
            f"Shiur ID {shiur_id} not found in the similarity matrix.")
        raise HTTPException(status_code=404, detail="Shiur ID not found")
    
@router.get("/cycle-recommendations-all")
def get_todays_recommendations(date=date.today()):
    return cycle_recommendations.get_all_recommendations(date)