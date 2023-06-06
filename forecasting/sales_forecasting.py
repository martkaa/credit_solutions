from models.xgboost import train_xgboost_model, forecast_xgboost_model

def forecast_sales(data, params):
    # Train the model
    model = train_xgboost_model(data, params)
    # Make the forecast
    sales_forecast = forecast_xgboost_model(model, params)
    return sales_forecast