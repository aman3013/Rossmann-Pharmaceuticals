import pandas as pd
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPipeline:
    def __init__(self, train_path, store_path, test_path):
        self.train_path = train_path
        self.store_path = store_path
        self.test_path = test_path
        self.merged_data = None
        self.train_data = None

    def load_data(self):
        logging.info("Loading data...")
        train = pd.read_csv(self.train_path)
        store = pd.read_csv(self.store_path)
        test = pd.read_csv(self.test_path)

        self.merged_data = pd.merge(train, store, on='Store', how='left')
        logging.info("Data loaded successfully.")
        return self.merged_data, test, store

    def clean_store_data(self, store_data):
        if store_data is None or store_data.empty:
            raise ValueError("Store data is empty or not provided.")
        
        logging.info("Cleaning store data...")
        store_data['CompetitionDistance'].fillna(0, inplace=True)
        store_data['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        store_data['CompetitionOpenSinceYear'].fillna(0, inplace=True)
        store_data['Promo2SinceWeek'].fillna(0, inplace=True)
        store_data['Promo2SinceYear'].fillna(0, inplace=True)
        store_data['PromoInterval'].fillna('None', inplace=True)
        logging.info("Store data cleaned successfully.")
        
        return store_data

    def clean_train_data(self):
        if self.merged_data is None:
            raise ValueError("Data not loaded. Please run load_data first.")
        logging.info("Cleaning train data...")
        
        # Assuming 'Sales' and 'Customers' are columns in self.merged_data
        self.train_data = self.merged_data.copy()
        
        # Example cleaning steps
        self.train_data.dropna(inplace=True)  # Adjust according to your cleaning needs
        
        logging.info("Train data cleaned successfully.")
        return self.train_data  # Return the cleaned DataFrame
    def clean_store_data(self, store_df):
        if store_df is None:
            raise ValueError("Store data not loaded.")
        logging.info("Cleaning store data...")
        # Your cleaning logic here
        self.store_data = store_df  # Set the cleaned store data
        logging.info("Store data cleaned successfully.")

    def detect_outliers(self):
        if self.merged_data is None:
            raise ValueError("Data not loaded. Please run load_data first.")
        
        logging.info("Detecting outliers...")
        self.merged_data['Sales_zscore'] = stats.zscore(self.merged_data['Sales'])
        self.merged_data['Customers_zscore'] = stats.zscore(self.merged_data['Customers'])
        
        # Remove outliers where Z-score is above 3 or below -3
        self.merged_data = self.merged_data[~((self.merged_data['Sales_zscore'] > 3) | 
                                               (self.merged_data['Sales_zscore'] < -3) |
                                               (self.merged_data['Customers_zscore'] > 3) | 
                                               (self.merged_data['Customers_zscore'] < -3))]
        
        self.merged_data.drop(columns=['Sales_zscore', 'Customers_zscore'], inplace=True)
        logging.info("Outliers detected and removed.")

    # Add visualization methods here
    # Example for plotting promo distribution
    def plot_promo_distribution(self, train_data, test_data):
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        sns.countplot(data=train_data, x='Promo')
        plt.title('Promo Distribution in Train Set')
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.countplot(data=test_data, x='Promo')
        plt.title('Promo Distribution in Test Set')
        plt.show()

    def sales_trend_by_month(self):
        if self.merged_data is None:
            raise ValueError("Data not loaded. Please run load_data first.")
        
        logging.info("Visualizing sales trend by month...")
        
        # Ensure 'Date' column is in datetime format
        self.merged_data['Date'] = pd.to_datetime(self.merged_data['Date'], errors='coerce')
        
        # Extract month from the date column
        self.merged_data['Month'] = self.merged_data['Date'].dt.month
        
        # Group data by month and sum up sales
        monthly_sales = self.merged_data.groupby('Month')['Sales'].sum().reset_index()
        
        # Plotting
        plt.figure(figsize=(10, 6))  # Ensure plt is the imported module
        sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o')
        plt.title('Sales Trend Across Months (Seasonality)')
        plt.show()
        
        logging.info("Sales trend visualization completed.")

    def sales_trend_by_month(self):
        if self.merged_data is None:
            raise ValueError("Data not loaded. Please run load_data first.")
        
        logging.info("Visualizing sales trend by month...")
        
        # Ensure 'Date' column is in datetime format
        self.merged_data['Date'] = pd.to_datetime(self.merged_data['Date'], errors='coerce')
        
        # Extract month from the date column
        self.merged_data['Month'] = self.merged_data['Date'].dt.month
        
        # Group data by month and sum up sales
        monthly_sales = self.merged_data.groupby('Month')['Sales'].sum().reset_index()
        
        # Plotting
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o')
        plt.title('Sales Trend Across Months (Seasonality)')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid()
        plt.show()
        
        logging.info("Sales trend visualization completed.")

    def sales_per_store_with_promos(self):
        if self.merged_data is None:
            raise ValueError("Data not loaded. Please run load_data first.")
        
        logging.info("Visualizing sales per store with and without promos...")
        
        # Check sales per store with and without promos
        store_promo_sales = self.merged_data.groupby(['Store', 'Promo'])['Sales'].sum().unstack()
        
        # Visualize the difference
        plt.figure(figsize=(12, 6))
        store_promo_sales.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Store-wise Sales with and without Promo')
        plt.xlabel('Store')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.legend(title='Promo', labels=['No Promo', 'With Promo'])
        plt.tight_layout()
        plt.show()
        
        logging.info("Sales per store visualization completed.")

    def sales_distribution_by_day(self):
        if self.merged_data is None:
            raise ValueError("Data not loaded. Please run load_data first.")
        
        logging.info("Visualizing sales distribution across days of the week...")
        
        # Group sales by stores open on weekdays vs weekends
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.merged_data, x='DayOfWeek', y='Sales')
        plt.title('Sales Distribution Across Days of the Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('Sales')
        plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.grid(axis='y')
        plt.show()
        
        logging.info("Sales distribution visualization completed.")

    def sales_by_assortment(self):
     if self.merged_data is None:
        raise ValueError("Data not loaded. Please run load_data first.")
     if self.store_data is None:
        raise ValueError("Store data not loaded. Please run load_store_data first.")
    
    # Merge store data with train data for assortment analysis
     merged_data = pd.merge(self.merged_data, self.store_data, on='Store')
    
    # Print columns for debugging
     print(merged_data.columns)
     print(merged_data.head())

    # Use Assortment_y or Assortment_x based on the merged data
     if 'Assortment_y' in merged_data.columns:
        assortment_col = 'Assortment_y'
     elif 'Assortment_x' in merged_data.columns:
        assortment_col = 'Assortment_x'
     else:
        raise ValueError("Assortment column not found in merged data.")

    # Visualize sales by assortment type
     plt.figure(figsize=(10, 6))
     sns.boxplot(data=merged_data, x=assortment_col, y='Sales')
     plt.title('Sales by Assortment Type')
     plt.show()
     logging.info("Sales by assortment type visualized.")

        

    def sales_vs_competitor_distance(self):
        if self.merged_data is None:
            raise ValueError("Data not loaded. Please run load_data first.")
        if self.store_data is None:
            raise ValueError("Store data not loaded. Please run load_store_data first.")

        # Merge store data with train data for competitor distance analysis
        merged_data = pd.merge(self.merged_data, self.store_data, on='Store')

        # Check if 'CompetitionDistance' exists
        if 'CompetitionDistance_y' in merged_data.columns:
            competition_distance_col = 'CompetitionDistance_y'
        elif 'CompetitionDistance_x' in merged_data.columns:
            competition_distance_col = 'CompetitionDistance_x'
        else:
            raise ValueError("CompetitionDistance column not found in merged data.")

        # Visualize sales vs competitor distance
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged_data, x=competition_distance_col, y='Sales')
        plt.title('Sales vs Competitor Distance')
        plt.xlabel('Competition Distance')
        plt.ylabel('Sales')
        plt.show()
        logging.info("Sales vs competitor distance visualized.")
