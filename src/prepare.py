import pandas as pd 
from preprocesssing import preprocessor

data = pd.read_csv("../data/new.csv")


X = data.drop(columns=["next_over_score"], axis=1)
y=data["next_over_score"]

num_col = X.select_dtypes(include=["int64","float64"]).columns
X= X[num_col]

pipe = preprocessor()

X_transformed = pipe.fit_transform(X)  #applying preprocessing at X 

X_df = pd.DataFrame(X_transformed)

 # Combine features + target
final_df = pd.concat([X_df, y.reset_index(drop=True)], axis=1)

 # Save processed data
final_df.to_csv("../data/processed_cricket.csv", index=False)
print("Preprocessed data saved to ../data/processed_cricket.csv")



