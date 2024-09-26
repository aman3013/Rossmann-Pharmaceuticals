import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from scipy import stats

# Path to scripts (optional if needed)
sys.path.append('../scripts')

# Function to load data
def load_data(train_path, store_path):
    train = pd.read_csv(train_path)
    store = pd.read_csv(store_path)
    df = pd.merge(train, store, on='Store', how='left')
    return df

# Function for preprocessing the dataset
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.dayofweek
    df['Is_weekend'] = (df['Weekday'] >= 5).astype(int)
    df['Month_part'] = pd.cut(df['Day'], bins=[0, 10, 20, 31], labels=[0, 1, 2])

    # Handle missing values
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].mean(), inplace=True)
    df['CompetitionOpenSinceMonth'].fillna(df['Month'], inplace=True)
    df['CompetitionOpenSinceYear'].fillna(df['Year'], inplace=True)
    df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: 0 if x < 0 else x)
    
    df['Promo2SinceWeek'].fillna(1, inplace=True)
    df['Promo2SinceYear'].fillna(df['Year'], inplace=True)
    df['PromoInterval'].fillna('', inplace=True)

    # Create a feature for Promo2
    df['Is_Promo2'] = df.apply(lambda x: 0 if x['Promo2'] == 0 else 1 if x['PromoInterval'] == '' else 1 if x['Month'] in x['PromoInterval'].split(',') else 0, axis=1)

    # Handle categorical variables
    df = pd.get_dummies(df, columns=['StoreType', 'Assortment', 'StateHoliday'])

    return df

# Function to extract features and target variable
def extract_features_and_target(df):
    features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Weekday', 'Is_weekend',
                'Month_part', 'Month', 'Year', 'CompetitionDistance', 'CompetitionOpen',
                'Promo2', 'Is_Promo2', 'Customers'] + [col for col in df.columns if col.startswith(('StoreType_', 'Assortment_', 'StateHoliday_'))]
    
    X = df[features]
    y = df['Sales']
    
    # Convert columns to numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    return X, y

# Function to train a RandomForestRegressor model
def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Function to evaluate the model
def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    return mae, rmse, y_pred

# Function to plot feature importance
def plot_feature_importance(pipeline, X_train):
    feature_names = X_train.columns.tolist()
    feature_importance = pipeline.named_steps['rf'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
    plt.title('Top 15 Most Important Features')
    plt.show()

# Function to save the model
def save_model(pipeline):
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    joblib.dump(pipeline, f'model_{timestamp}.pkl')

# Function to predict with confidence intervals
def predict_with_confidence(model, X, confidence=0.95):
    n_iterations = 100
    predictions = np.zeros((n_iterations,) + (X.shape[0],))
    
    for i in range(n_iterations):
        predictions[i,:] = model.predict(X)
    
    y_pred = np.mean(predictions, axis=0)
    y_err = stats.t.ppf((1 + confidence) / 2., n_iterations - 1) * np.std(predictions, axis=0)
    
    return y_pred, y_err

# Function to plot predictions with confidence intervals
def plot_predictions_with_confidence(y_test, y_pred, y_err):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.fill_between(range(len(y_pred)), y_pred - y_err, y_pred + y_err, alpha=0.2)
    plt.title('Predicted vs Actual Sales with Confidence Intervals')
    plt.xlabel('Sample')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

# Main function to orchestrate the process
def main():
    # Load data
    df = load_data('../data/train.csv', '../data/store.csv')
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Extract features and target variable
    X, y = extract_features_and_target(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    pipeline = train_model(X_train, y_train)
    
    # Evaluate the model
    mae, rmse, y_pred = evaluate_model(pipeline, X_test, y_test)
    
    # Plot feature importance
    plot_feature_importance(pipeline, X_train)
    
    # Predict with confidence intervals
    y_pred_conf, y_err = predict_with_confidence(pipeline, X_test)
    
    # Plot predictions with confidence intervals
    plot_predictions_with_confidence(y_test, y_pred_conf, y_err)
    
    # Save the model
    save_model(pipeline)

# Run the main function
if __name__ == "__main__":
    main()
