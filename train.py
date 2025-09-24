import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from src.triedmodel import get_models
from src.evaluate import evaluate_model
import joblib


data = pd.read_csv("data/processed_cricket.csv")

X= data.drop("next_over_score", axis=1)
y= data["next_over_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = get_models()


results= []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #to evaluate
    metrics = evaluate_model(y_test, y_pred)
    metrics["model"]= name
    results.append(metrics)

df_results = pd.DataFrame(results).sort_values(by="r2_score", ascending=True)
print("\nModel Comparison:\n", df_results)

joblib.dump(models["Random forest regressor"],"Final_model/rf.pkl")
joblib.dump(models["XGBRegressor"],"Final_model/xgb.pkl")
joblib.dump(models["LGBMRegressor"],"Final_model/lgbm.pkl")
joblib.dump(models["CatBoostRegressor"],"Final_model/cat.pkl")

print("all model saved successfully in Final_models folder")