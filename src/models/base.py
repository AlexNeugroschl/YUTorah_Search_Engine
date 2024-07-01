from typing import List, Dict


class BaseModel():
    # List of shiurIDs
    def get_recommendations(self, user_id: str = None, *args, **kwargs) -> List[int]:
        raise NotImplementedError(
            "This method should be overridden by subclasses")

    # Dict of {shiurIDs: recommend_probability}
    def get_weighted_recommendations(self, user_id: str = None, *args, **kwargs) -> Dict[int, float]:
        raise NotImplementedError(
            "This method should be overridden by subclasses")
