import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
df=pd.read_csv('2021_Green_Taxi_Trip_Data.csv')
df.describe()
irrelevant_columns = ['RatecodeID', 'store_and_fwd_flag', 'improvement_surcharge',
                    'ehail_fee', 'mta_tax', 'extra', 'fare_amount',
                    'congestion_surcharge']

# Select relevant columns, handling possible missing columns
df = (df.drop(irrelevant_columns,axis=1))
print(df.columns)
df['payment_type'] = df['payment_type'].astype(object)
df['trip_type'] = df['trip_type'].astype(object)
df.loc[(df['tip_amount'] > 0) & (df['payment_type'].isnull()), 'payment_type'] = 1
df.loc[df['passenger_count'] > 4, 'payment_type'] = 'unknown'
df.loc[df['passenger_count'] > 4, 'trip_type'] = 'dispatch'
df.dropna(subset=['VendorID'], inplace=True)
df['pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df = df[df['pickup_datetime'].dt.year == 2021]
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
print(missing_value_df)


