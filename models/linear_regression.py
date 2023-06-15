def seasonally_adjust(time_series):
    # Apply seasonal adjustment to the time series
    return seasonal_decompose(time_series, model='multiplicative', freq=7)