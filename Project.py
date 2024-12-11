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
sns.barplot(data=trip_type_hourly_distribution.reset_index().melt(id_vars=['pickup_hour']),
            x='pickup_hour', y='value', hue='trip_type', ci=None)
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
sns.barplot(data=payment_type_hourly_distribution.reset_index().melt(id_vars=['pickup_hour']),
            x='pickup_hour', y='value', hue='payment_type', ci=None)
plt.title('Payment Type Distribution by Hour of Day')  
plt.xlabel('Hour of Day')  
plt.ylabel('Number of Trips')  
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability  
plt.legend(title='Payment Type', bbox_to_anchor=(1.05, 1), loc='upper left')  
plt.tight_layout()  
plt.show()  

# 3. حذف سفرهایی که نوع آن‌ها dispatch است
taxi_df = df[df['trip_type'] != 'dispatch']
# بررسی استفاده از تاکسی سبز بر اساس بخش‌های شهری فقط برای تاکسی‌ها
green_taxi_usage = taxi_df.groupby('pickup_hour').size()

# رسم نمودار
plt.figure(figsize=(10, 6))
green_taxi_usage.plot(kind='bar', color='green')
plt.title('Green Taxi Usage by Hour of Day ')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Green Taxi Trips')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# 4
# borough_df = pd.read_csv('Boroughs.csv')  
# print(borough_df.head())  
# print(borough_df.info())  

# # Merge borough information into the main DataFrame  
# # Assuming Borough.csv has 'LocationID' and 'Borough' columns  
# df = df.merge(borough_df[['LocationID', 'Borough']], left_on='PULocationID', right_on='LocationID', how='left')  
# df.rename(columns={'Borough': 'PU_Borough'}, inplace=True)  # Rename the column for pickup borough  

# df = df.merge(borough_df[['LocationID', 'Borough']], left_on='DOLocationID', right_on='LocationID', how='left')  
# df.rename(columns={'Borough': 'DO_Borough'}, inplace=True)  # Rename the column for drop-off borough  


# 4. بارگذاری داده‌ها
df = pd.read_csv('2021_Green_Taxi_Trip_Data.csv')

# تبدیل به فرمت datetime
df['pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
df['pickup_hour'] = df['pickup_datetime'].dt.hour

# گروه‌بندی بر اساس مناطق شهری
borough_mapping = {
    1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn', 4: 'Queens', 5: 'Staten Island'
}
df['borough'] = df['PULocationID'].map(borough_mapping)

# شمارش تعداد سفرها در هر منطقه
usage_by_borough = df.groupby(['borough', 'pickup_hour']).size().unstack(fill_value=0)

# رسم نمودار
usage_by_borough.T.plot(figsize=(14, 7), kind='bar')
plt.title('Green Taxi Usage Trend by Borough')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Trips')
plt.legend(title='Borough', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

##############            بخش دوم
# محاسبه ماتریس همبستگی
correlation_matrix = df.corr()
# رسم ماتریس همبستگی به صورت Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix for Predicting Total Amount')
plt.show()

# تحلیل: ستون‌هایی که بیشترین همبستگی را با total_amount دارند
correlation_with_total = correlation_matrix['total_amount'].sort_values(ascending=False)
print("Top correlations with total_amount:")
print(correlation_with_total)


