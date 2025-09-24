from  sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor 
from catboost import  CatBoostRegressor


def get_models():
    return{
        "Random forest regressor":RandomForestRegressor( n_estimators=100, max_depth=None, random_state=42, n_jobs=-1),
        "XGBRegressor":XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42),
        "LGBMRegressor":LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=7, random_state=42, verbose=-1),
        "CatBoostRegressor":CatBoostRegressor(iterations= 100, learning_rate= 0.1, depth= 6, l2_leaf_reg= 3, random_seed= 42, verbose= False),
    }