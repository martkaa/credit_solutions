def train_xgboost_model(df, params):
    # Create the XGBoost data structure
    dtrain = xgb.DMatrix(df[params['features']], label=df[params['label']])
    # Train the model
    model = xgb.train(params['xgboost_params'], dtrain, params['num_rounds'])
    return model

def forecast_xgboost_model(model, df, params):
    # Create the XGBoost data structure
    dtest = xgb.DMatrix(df[params['features']])
    # Make the forecast
    forecast = model.predict(dtest)
    return forecast