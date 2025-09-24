from sklearn.pipeline import  Pipeline
from sklearn.impute import SimpleImputer

def preprocessor():
    num_pipe = Pipeline([
        ("handle_missing_val",SimpleImputer(strategy="mean"))
    ])


    return num_pipe
