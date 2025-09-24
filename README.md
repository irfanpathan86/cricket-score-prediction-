 Cricket Score Prediction

This project predicts the total cricket score of an inning using machine learning models. The prediction is based on the first 10 overs of input data and predicts the scores for the 11th to 20th overs. It then calculates the total score for the inning.


Features

- Predicts scores for each over from 11th to 20th based on the first 10 overs.
- Uses multiple regression models:
  - XGBoost
  - Random Forest
  - LightGBM
  - CatBoost
- Calculates the total score combining actual first 10 overs and predicted remaining overs.
- Progressive prediction: Each new over prediction considers the previous predicted overs.

Installation

1. Clone the repository:

cd cricket-score-prediction-
python -m venv venv
# for window 
.\venv\Scripts\activate
# for linux/ mac 
source venv/bin/activate


