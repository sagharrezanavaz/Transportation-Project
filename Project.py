import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  

# Load the dataset  
df = pd.read_csv('2021_Green_Taxi_Trip_Data.csv')  

# Drop irrelevant columns  
irrelevant_columns = ['RatecodeID', 'store_and_fwd_flag', 'improvement_surcharge',  
                      'ehail_fee', 'mta_tax', 'extra', 'fare_amount',  
                      'congestion_surcharge']  
df = df.drop(irrelevant_columns, axis=1)  

# Convert relevant columns to appropriate types  
df['payment_type'] = df['payment_type'].astype(object)  
df['trip_type'] = df['trip_type'].astype(object)  

# Handle specific missing values  
df.loc[(df['tip_amount'] > 0) & (df['payment_type'].isnull()), 'payment_type'] = 1  
df.loc[df['passenger_count'] > 4, 'payment_type'] = 'unknown'  
df.loc[df['passenger_count'] > 4, 'trip_type'] = 'dispatch'  
df.dropna(subset=['VendorID', 'passenger_count'], inplace=True)  

# Convert pickup datetime  
df['pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')  
df = df[df['pickup_datetime'].dt.year == 2021]  

# Extract hour from pickup_datetime  
df['pickup_hour'] = df['pickup_datetime'].dt.hour  

# 1. Trip Type Distribution by Hour  
trip_type_hourly_distribution = df.groupby(['pickup_hour', 'trip_type']).size().unstack(fill_value=0)  

# Plotting the Trip Type Distribution by Hour  
plt.figure(figsize=(14, 7))  
trip_type_hourly_distribution.plot(kind='bar', stacked=True)  
plt.title('Trip Type Distribution by Hour of Day')  
plt.xlabel('Hour of Day')  
plt.ylabel('Number of Trips')  
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability  
plt.legend(title='Trip Type', bbox_to_anchor=(1.05, 1), loc='upper left')  
plt.tight_layout()  
plt.show()  

# 2. Payment Type Distribution by Hour  
payment_type_hourly_distribution = df.groupby(['pickup_hour', 'payment_type']).size().unstack(fill_value=0)  

# Plotting the Payment Type Distribution by Hour  
plt.figure(figsize=(14, 7))  
payment_type_hourly_distribution.plot(kind='bar', stacked=True)  
plt.title('Payment Type Distribution by Hour of Day')  
plt.xlabel('Hour of Day')  
plt.ylabel('Number of Trips')  
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability  
plt.legend(title='Payment Type', bbox_to_anchor=(1.05, 1), loc='upper left')  
plt.tight_layout()  
plt.show()  
borough_df = pd.read_csv('Boroughs.csv')  
print(borough_df.head())  
print(borough_df.info())  

# Merge borough information into the main DataFrame  
# Assuming Borough.csv has 'LocationID' and 'Borough' columns  
df = df.merge(borough_df[['LocationID', 'Borough']], left_on='PULocationID', right_on='LocationID', how='left')  
df.rename(columns={'Borough': 'PU_Borough'}, inplace=True)  # Rename the column for pickup borough  

df = df.merge(borough_df[['LocationID', 'Borough']], left_on='DOLocationID', right_on='LocationID', how='left')  
df.rename(columns={'Borough': 'DO_Borough'}, inplace=True)  # Rename the column for drop-off borough  

# Check the DataFrame to ensure it has the expected structure  
print(df.head(20))  
print(df.info()) 



