import pandas as pd
import joblib
#sample code for pridicting the score of new data of 10 over from Book1 
models = {
    "XGBoost": joblib.load("Final_model/xgb.pkl"),
    "RandomForest": joblib.load("Final_model/rf.pkl"),
    "LightGBM": joblib.load("Final_model/lgbm.pkl"),
    "CatBoost": joblib.load("Final_model/cat.pkl")
}

user_input = pd.read_csv("Book1.csv")  
predicted_df = pd.DataFrame(user_input.copy())
for model_name in models.keys():
    predicted_df[model_name] = None

actual_first_10_scores = user_input.iloc[:, -1].tolist() 


total_overs = 20
for model_name, model in models.items():
    current_data = user_input.copy()
    
    for over in range(len(user_input), total_overs):
        last_row = current_data.iloc[[-1]].copy()
        predicted_score = float(model.predict(last_row)[0])
        predicted_df.loc[over, model_name] = predicted_score
        
        
        current_data = pd.concat([current_data, last_row], ignore_index=True)

sum_first_10 = sum(actual_first_10_scores)
total_scores = {}
for model_name in models.keys():
    predicted_sum = predicted_df[model_name][10:].sum()
    total_scores[model_name] = {
        "sum_first_10": sum_first_10,
        "sum_predicted_11_20": predicted_sum,
        "total_score": sum_first_10 + predicted_sum
    }

print("\nPredicted overs 11–20 per model:\n")
print(predicted_df[10:][list(models.keys())])

print("\nTotal score details per model:\n")
for model, scores in total_scores.items():
    print(f"{model}:")
    print(f"  Sum of first 10 overs: {scores['sum_first_10']:.2f}")
    print(f"  Sum of predicted overs 11–20: {scores['sum_predicted_11_20']:.2f}")
    print(f"  Total score: {scores['total_score']:.2f}\n")
