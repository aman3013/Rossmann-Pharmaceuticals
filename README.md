# Rossmann-Pharmaceuticals

## Overview

This project focuses on building an end-to-end solution to predict store sales at Rossmann Pharmaceuticals. The finance team requires a model to forecast sales for all stores six weeks ahead of time, considering factors such as promotions, competition, holidays, and seasonality. This will enable better planning and resource allocation for the company.

## Project Workflow
The project is divided into three main tasks:

1. **Exploration of Customer Purchasing Behavior (EDA)**
2. **Prediction of Store Sales using Machine Learning**
3. **Model Serving via REST API**

## Data
The dataset used for this project is available on [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data). It includes the following fields:

- **Id**: (Store, Date) tuple identifier.
- **Store**: Unique identifier for each store.
- **Sales**: Target variable representing store turnover.
- **Customers**: Number of customers on a given day.
- **Open**: Indicator for store opening (0: closed, 1: open).
- **StateHoliday**: Indicates state holidays (a: public holiday, b: Easter, c: Christmas, 0: None).
- **SchoolHoliday**: Indicates whether a store was affected by public school closures.
- **StoreType**: Four types of stores: a, b, c, d.
- **Assortment**: Assortment level (a: basic, b: extra, c: extended).
- **CompetitionDistance**: Distance to the nearest competitor store.
- **Promo/Promo2**: Promotional indicators.
- **Promo2SinceYear/Week**: Year/Week when Promo2 started.
- **PromoInterval**: The months in which Promo2 runs.
  

## Task 1: Exploration of Customer Purchasing Behavior (EDA)

### Goals
- Understand customer behavior across stores.
- Analyze the effects of holidays, promotions, and store types on sales.
- Investigate competition distance and how it impacts store performance.

### Key Questions
- Are promotions distributed similarly between training and test sets?
- How do holidays affect sales?
- Are there seasonal patterns in sales?
- What is the correlation between sales and the number of customers?
- How do promotions affect existing and new customers?
- How does store type and assortment influence sales?

### Data Cleaning & Visualization
- **Data Preprocessing**: Detect and handle outliers and missing data.
- **Feature Exploration**: Use plots to visualize feature interactions and customer behavior trends.

### Logging
Log all steps using the `logging` library to ensure traceability and reproducibility of analysis.


## Task 2: Prediction of Store Sales

### 2.1 Data Preprocessing
- **Feature Engineering**: Extract new features such as weekdays, weekends, holiday proximity, month segments, etc.
- **Handling Missing Data**: Impute missing values appropriately.
- **Scaling**: Use `StandardScaler` from scikit-learn to scale data for better model performance.

### 2.2 Machine Learning Model
- **Model Selection**: Tree-based algorithms such as Random Forest Regressor.
- **Pipeline Design**: Build a machine learning pipeline using scikit-learn for modularity and reproducibility.

### 2.3 Loss Function
- Choose and defend the selection of a loss function that best fits the business need.

### 2.4 Post Prediction Analysis
- **Feature Importance**: Explore key features affecting model performance.
- **Confidence Interval**: Estimate prediction confidence intervals.

### 2.5 Model Serialization
- Save trained models with a timestamp (e.g., `model_2024-10-12_16-32.pkl`) for future use in real-time predictions.

### 2.6 Deep Learning Approach
- Build a simple 2-layer Long Short-Term Memory (LSTM) model using TensorFlow or PyTorch.
- Convert time-series sales data into supervised learning data and train the LSTM to predict future sales.


## Task 3: Model Serving API

### REST API for Real-time Predictions
- **Framework**: Use Flask or FastAPI to build the REST API.
- **Model Loading**: Load the serialized model from Task 2.
- **API Endpoints**:
  - POST `/predict`: Receives input data and returns sales predictions.
- **Preprocessing**: Handle input preprocessing within the API.
- **Deployment**: Deploy the API to a cloud platform (e.g., AWS, Heroku).


## Learning Outcomes

### Skills:
- Advanced use of Python (Pandas, Numpy, Matplotlib, Scikit-learn).
- Feature Engineering and Model Building.
- CI/CD for ML models (MLOps, DVC, MLFlow).
- Deployment of ML models via REST APIs.
- Logging and unit testing.
  
### Knowledge:
- Predictive analysis and forecasting.
- Business context reasoning.
- Machine Learning and Hyperparameter Tuning.
- Model Comparison & Selection.

### Communication:
- Reporting complex statistical findings.
- Creating dashboards and visual reports for stakeholders.

## Installation

### Prerequisites
- Python 3.8+
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, TensorFlow/PyTorch, Flask/FastAPI.

### Install the required packages:

```bash
pip install -r requirements.txt
python eda.py
python train_model.py
python app.py
|-- data/                     # Raw and processed data
|-- notebooks/                # Jupyter notebooks for EDA and modeling
|-- models/                   # Serialized machine learning models
|-- src/                      # Source code
|   |-- eda.py                # EDA script
|   |-- train_model.py        # Model training script
|   |-- app.py                # REST API server script
|-- logs/                     # Logging information
|-- requirements.txt          # Python dependencies
|-- README.md                 # Project documentation

