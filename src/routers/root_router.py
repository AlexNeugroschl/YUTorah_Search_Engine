from fastapi import APIRouter, HTTPException
from typing import Dict
from ..logging_config import setup_logging
from src.models.content_handler_v2 import ContentHandler

router = APIRouter()
content_filtering = ContentHandler()

logger = setup_logging()


@router.get("/content-recommendations/{user_id}", response_model=Dict[int, str])
def get_content_recommendations(user_id: int, top_n: int = 10):
    try:
        recommendations = content_filtering.recommend_for_user_content(
            user_id, top_n)
        return recommendations
    except KeyError:
        logger.error(
            f"Shiur ID {user_id} not found in the similarity matrix.")
        raise HTTPException(status_code=404, detail="Shiur ID not found")


@router.get("/becuase-you-listened-recommendations/{user_id}", response_model=Dict[int, str])
def get_because_you_listened_recommendations(user_id: int, top_n: int = 5):
    try:
        recommendations = content_filtering.recommend_based_on_recent_activity(
            user_id, top_n)
        return recommendations
    except KeyError:
        logger.error(
            f"Shiur ID {user_id} not found in the similarity matrix.")
        raise HTTPException(status_code=404, detail="Shiur ID not found")
