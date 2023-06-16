# CLV calculation

Credit Solutions is a project designed to predict the Customer Lifetime Value (CLV) of clients. This model is particularly useful for credit companies who need to analyze and make decisions based on the creditworthiness and value of their customers. The project includes data preparation, exploratory analysis, time series analysis, feature engineering, model selection, forecasting, and calculating the CLV for each customer. It also aims at exploring the potential applications of this information.



## Table of Contents
- [CLV Theoretical Introduction](#CLV-th)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Pipeline](#pipeline)
- [Future Work](#future-work)

## CLV Theoretical Introduction

The CLV calculation is based on the following formula for customer lifetime value. This formula is the mathematical representation of the CLV of all of a comanpy's customers in the context of a credit extender.

$$
CLV_{N_{t},t} = \sum_{n=1}^{E_{t}(N_{t})} \sum_{t=0}^{E_{t}(T_{i,t})} \frac{E_{t}\left [ CF_{i,t} \right ]-E_{t|default}\left [ r \cdot D_{i,t} \right ]}{(1+k_{i,t})^{t}}
$$

- $E_{t}(N_{t})$: The expected number of customers at time \(t\).
- $E_{t}(T_{i,t})$: The expected future duration \(T\) of customer \(i\) with the company from time \(t\).
- $E_{t}\left[ CF_{i,t} \right]$: The expected net cash flow for customer \(i\) at time \(t\), which may be decomposed into expected sales times expected profitability.
- $E_{t|d}\left[ r \cdot D_{i,t} \right]$: The net exposure with respect to customer \(i\) at time \(t\) given customer default.
- $(1+k_{i,t})^{t}$: The discount factor \(k\) for customer \(i\) at time \(t\) which is used to calculate the present value of future cash flows.
- $k_{i,t}$: The risk-adjusted discount rate, which represents the expected required rate of return for customer \(i\) at time \(t\).

The CLV for one customer is thus given by the following equation and is what is attempted implemneted in the application.

$$
CLV_{i,t} = \sum_{t=0}^{E_{t}(T_{i,t})} \frac{E_{t}\left [ CF_{i,t} \right ]-E_{t|default}\left [ r \cdot D_{i,t} \right ]}{(1+k_{i,t})^{t}}
$$


## Installation

**Requirements:**

- Python 3.7+
- Pandas
- Numpy
- scikit-learn
- statsmodels
- pmdarima
- matplotlib

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip install pandas numpy scikit-learn statsmodels pmdarima matplotlib
```

## Modules

- **data_processing**: Contains the pre-processing logic like data cleaning and transformation.
- **models**: Contains various forecasting models, including ARIMA, Random Forest, XGBoost, Linear Regression, and Cox Proportional Hazard Risk
- **forecasting**: Responsible for time series analysis and forecasting.
- **customer**: Contains the Customer class which consolidates the various modules to compute the CLV.

## Pipeline

1. **Data Preparation**
    - Data Collection
    - Data Cleaning
    - Data Transformation

2. **Exploratory Data Analysis**
    - Understanding Patterns
    - Visualization

3. **Time Series Analysis**
    - Checking for Stationarity, removing stationarity
    - Checking for Seasonality, removing seasonality

4. **Feature Engineering**
    - Engineering Relevant Features
    - Feature Selection

5. **Model Selection**
    - Data Splitting
    - Model Training
    - Model Evaluation
    - Model Selection

6. **Forecasting**
    - Forecasting sales, profitability, and net exposure

7. **Forecasting customer duration**
    - Calculating time until default using aurvival analysis

8. **Forecasting RADR**
    - Using credit score to forecast RADR

9. **Calculating CLV**
    - Calculating the Customer Lifetime Value for each customer based on the forecasted elements
   
10. **Business Applications**
    - Dynamic CLV Visualization
    - Data-driven Decision Making (Adjusting credit limits, Risk management)
    - Potential for Automation (Establishing and monitoring credit limits, Customer classification)

## Future Work

- Developing segmentation rules based on CLV.
- Building a dynamic reporting dashboard.
- Automating the establishment and monitoring of credit limits.

## Data Requirements and Considerations

- Ensure data quality and integrity.
- Consider data sourcing and handling strategies.