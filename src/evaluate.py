from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_squared_error

def evaluate_model(y_test, y_pred):
    return{
        "mse":mean_squared_error(y_test,y_pred),
        "rmse":root_mean_squared_error(y_test, y_pred),
        "r2_score":r2_score(y_test, y_pred),
        "mae":mean_absolute_error(y_test, y_pred)
    }
    
