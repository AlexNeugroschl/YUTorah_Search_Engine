import sys
import os
# Add the src directory to the Python path
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()  # Fallback to the current working directory
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(src_dir)
from src.pipeline.etl import ETL
from fastai.vision.all import *
from typing import Dict, List
from base import BaseModel
from fastai.collab import *



db = ETL()
df = db.get_bookmarks_df()
df['rating'] = 1
dls = CollabDataLoaders.from_df(df, user_name='user', item_name='shiur', rating_name='rating', bs=64)
learn = collab_learner(dls, n_factors=5, y_range=(0, 1), loss_func=BCEWithLogitsLossFlat())
learn.fit_one_cycle(15, 5e-3)

learn.model_dir = "saved_models/"
learn.save("user_collab_filtering_v1")


class UserCollabFilteringV1(BaseModel):
    def __init__(self):
        model = learn.load("user_collab_filtering_v1")

    def get_recommendations(self, user_id: str = None, *args, **kwargs) -> List[int]:
        top_n = kwargs.get('top_n', 10)
        user_id = int(float(user_id))
        item_ids = dls.classes['shiur'].items[1:] # to avoid the na value
        item_ids = [int(item_id) for item_id in item_ids]
        user_tensor = torch.tensor([user_id] * len(item_ids)).unsqueeze(1)
        item_tensor = torch.tensor(item_ids).unsqueeze(1)
        input_tensor = torch.cat((user_tensor, item_tensor), dim=1)
        
        # Get predictions
        input_df = pd.DataFrame(input_tensor.numpy(), columns=['user', 'shiur'])
        user_item_dl = dls.test_dl(input_df)
        preds, _ = learn.get_preds(dl=user_item_dl)
        
        # Get top N recommendations
        top_indices = torch.argsort(preds, descending=True)[:top_n]
        top_item_ids = [item_ids[idx.item()] for idx in top_indices]
        return top_item_ids

    def get_weighted_recommendations(self, user_id: str = None, *args, **kwargs) -> Dict[int, float]:
        top_n = kwargs.get('top_n', 10)
        user_id = int(float(user_id))
        item_ids = dls.classes['shiur'].items[1:]
        item_ids = [int(item_id) for item_id in item_ids]
        user_tensor = torch.tensor([user_id] * len(item_ids)).unsqueeze(1)
        item_tensor = torch.tensor(item_ids).unsqueeze(1)
        input_tensor = torch.cat((user_tensor, item_tensor), dim=1)

        # Get predictions
        input_df = pd.DataFrame(input_tensor.numpy(), columns=['user', 'shiur'])
        user_item_dl = dls.test_dl(input_df)
        preds, _ = learn.get_preds(dl=user_item_dl)

        # Get top N recommendations with their scores
        top_indices = torch.argsort(preds, descending=True)[:top_n]
        top_item_ids = [item_ids[idx.item()] for idx in top_indices]
        top_scores = preds[top_indices].tolist()

        recommendations = {item_id: score for item_id, score in zip(top_item_ids, top_scores)}
        return recommendations

    def get_best_shiurim(self, shiur_num:int = 10): #based on highest bias
        shiur_bias = learn.model.i_bias.weight.squeeze()
        idxs = shiur_bias.argsort(descending=True)[:shiur_num]
        return [dls.classes['shiur'][i] for i in idxs]
    
    def get_user_bias(self, user_id:str = None):
        user_biases = learn.model.u_bias.weight
        user_idx = learn.dls.classes['user'].o2i[user_id]
        return user_biases[user_idx]
    
    def get_shiur_bias(self, shiur_id:str = None):
        item_biases = learn.model.i_bias.weight
        item_idx = learn.dls.classes['user'].o2i[shiur_id]
        return item_biases[item_idx]