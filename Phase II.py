import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
df = pd.read_csv('2021_Green_Taxi_Trip_Data.csv')

irrelevant_columns = ['RatecodeID', 'store_and_fwd_flag', 'improvement_surcharge',
                      'ehail_fee', 'mta_tax', 'extra', 'fare_amount',
                      'congestion_surcharge']
df = df.drop(irrelevant_columns, axis=1)
df.info()
df=df.drop_duplicates()
df = df[(df.trip_distance >= 0) & (df.tip_amount >= 0) & (df.tolls_amount >= 0) & (df.total_amount >= 0)]
df['payment_type'] = df['payment_type'].astype(object)
df['trip_type'] = df['trip_type'].astype(object)


df.loc[(df['tip_amount'] > 0) & (df['payment_type'].isnull()), 'payment_type'] = 1
df.loc[df['passenger_count'] > 4, 'payment_type'] = 6
df.loc[df['passenger_count'] > 4, 'trip_type'] = 3

empty_vendor_trips = df[df['VendorID'].isnull()].groupby(['PULocationID', 'DOLocationID']).size().unstack(fill_value=0)  
#print(empty_vendor_trips)
df.dropna(subset=['VendorID'], inplace=True)

df['pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df = df[df['pickup_datetime'].dt.year == 2021]
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
print(missing_value_df)

df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_day'] = df['pickup_datetime'].dt.day
df['pickup_month'] = df['pickup_datetime'].dt.month
df['weekday'] = df['pickup_datetime'].dt.day_name()
weekday_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}
# Apply the mapping to the 'weekday' column
df['weekday'] = df['weekday'].map(weekday_mapping)


df['dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df['dropoff_hour'] = df['dropoff_datetime'].dt.hour
df['dropoff_day'] = df['dropoff_datetime'].dt.day
df['dropoff_month'] = df['dropoff_datetime'].dt.month


df.drop(columns=['lpep_pickup_datetime', 'lpep_dropoff_datetime'], inplace=True)
borough_df = pd.read_csv('Boroughs.csv')
df = df.merge(borough_df[['LocationID', 'Borough']], left_on='PULocationID', right_on='LocationID', how='left')
df.rename(columns={'Borough': 'PU_Borough'}, inplace=True)

df = df.merge(borough_df[['LocationID', 'Borough']], left_on='DOLocationID', right_on='LocationID', how='left')
df.rename(columns={'Borough': 'DO_Borough'}, inplace=True)
df = pd.get_dummies(df, columns=['PU_Borough','DO_Borough'], prefix=['PU','DO'], drop_first=True)
df.drop([ 'PULocationID', 'DOLocationID', 'pickup_datetime', 'dropoff_datetime' ,], axis=1, inplace=True)
print(df.columns)
df = df.apply(pd.to_numeric, errors='coerce')
for column in df.columns:
    null_percentage = df[column].isnull().mean()
    if 0.05 < null_percentage <= 0.3:
        missing_rows = df[df[column].isnull()]
        complete_rows = df[df[column].notnull()]

        features_to_drop = [column]
        if complete_rows[column].dtype in ['int64', 'float64']:
            model = HistGradientBoostingRegressor()
            model.fit(complete_rows.drop(columns=features_to_drop, errors='ignore'), complete_rows[column])
            predicted_values = model.predict(missing_rows.drop(columns=features_to_drop, errors='ignore'))
            df.loc[df[column].isnull(), column] = predicted_values


        else:
            model = HistGradientBoostingClassifier()
            model.fit(complete_rows.drop(columns=features_to_drop, errors='ignore'), complete_rows[column])
            predicted_values = model.predict(missing_rows.drop(columns=features_to_drop, errors='ignore'))
            df.loc[df[column].isnull(), column] = predicted_values

print(df.LocationID_x.head())
print(df.LocationID_y.head())
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
print(missing_value_df)

df['PU_Bronx'] = (df.filter(like='PU_').sum(axis=1) == 0).astype(int)

df['DO_Bronx'] = (df.filter(like='DO_').sum(axis=1) == 0).astype(int)
# Create OD matrix
od_matrix = pd.DataFrame(index=df.columns[df.columns.str.startswith('PU_')],
                         columns=df.columns[df.columns.str.startswith('DO_')]).fillna(0)
for pu_col in df.columns[df.columns.str.startswith('PU_')]:
    for do_col in df.columns[df.columns.str.startswith('DO_')]:
        od_matrix.loc[pu_col, do_col] = df[(df[pu_col] == 1) & (df[do_col] == 1)].shape[0]

# Calculate sums for origins and destinations
od_matrix.loc['Total'] = od_matrix.sum()  # Sum for each drop-off column
od_matrix['Total'] = od_matrix.sum(axis=1)  # Sum for each pickup row

# Plotting the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(od_matrix, cmap="YlGnBu", annot=True, fmt=".0f", cbar_kws={"label": "Number of Trips"})
plt.title("Origin-Destination Matrix Heatmap")
plt.xlabel("Drop-off Borough")
plt.ylabel("Pickup Borough")
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.show()
################################
od_matrix = pd.DataFrame(index=df.columns[df.columns.str.startswith('PU_')],  
                         columns=df.columns[df.columns.str.startswith('DO_')]).fillna(0)  
for pu_col in df.columns[df.columns.str.startswith('PU_')]:  
    for do_col in df.columns[df.columns.str.startswith('DO_')]:  
        od_matrix.loc[pu_col, do_col] = df[(df[pu_col] == 1) & (df[do_col] == 1)].shape[0]  

# محاسبه مجموع کل سفرها  
total_trips = od_matrix.sum().sum()  # مجموع تمامی سفرها در ماتریس  

# محاسبه درصد سفرها در ماتریس  
od_matrix_percent = od_matrix.copy()  
for pu_col in od_matrix_percent.index:  
    for do_col in od_matrix_percent.columns:  
        if od_matrix_percent.loc[pu_col, do_col] > 0:  
            percent = (od_matrix_percent.loc[pu_col, do_col] / total_trips) * 100  
            od_matrix_percent.loc[pu_col, do_col] = percent  
        else:  
            od_matrix_percent.loc[pu_col, do_col] = 0  
# دیکشنری برای نگه‌داری ارتباط بین نام بوره‌ها و شناسه‌هایشان  
borough_ids = {  
    'Bronx': 1,  
    'Brooklyn': 2,  
    'Manhattan': 3,  
    'Queens': 4,  
    'Staten Island': 5  
}  

# ایجاد یک دیکشنری برای جمع‌آوری اطلاعات  
result = []  
for pu_col in od_matrix_percent.index:  
    for do_col in od_matrix_percent.columns:  
        percent = od_matrix_percent.loc[pu_col, do_col]  
        count = od_matrix.loc[pu_col, do_col]  # تعداد سفرها از ماتریس اصلی  
        if percent > 0:  # تنها جفت‌های مثبت  
            pickup_borough = pu_col.split('_')[-1]  # نام بوره  
            dropoff_borough = do_col.split('_')[-1]  # نام بوره  
            
            # ذخیره تمام مقادیر در نتیجه بدون اعمال شرط
            result.append({  
                'Pickup': pickup_borough,  
                'Dropoff': dropoff_borough,  
                'Percentage': percent,  
                'Trip Count': count  # استفاده از دیکشنری برای شناسه  
            })  

# تبدیل به DataFrame
result_df = pd.DataFrame(result)  

# اعمال شرط در سطح DataFrame: فیلتر کردن بخش‌هایی که مبدا و مقصد یکی هستند
result_diff = result_df[result_df['Pickup'] != result_df['Dropoff']]
result_diff = result_diff.sort_values(by='Percentage', ascending=False)  
result_diff.columns = ['Pickup Borough', 'Dropoff Borough', 'Percentage of Trips', 'Number of Trips']  
print("Ranking City Blocks by the Highest Percentage of Inter-Block Trips")
result_diff.index = np.arange(1, len(result_diff)+1)
print(result_diff.head(10))

result_same = result_df[result_df['Pickup'] == result_df['Dropoff']]
result_same = result_same.sort_values(by='Percentage', ascending=False)  
result_same.columns = ['Pickup Borough', 'Dropoff Borough', 'Percentage of Trips', 'Number of Trips']  
print("\n Ranking City Blocks by the Highest Percentage of Intra-Block Trips")
result_same.index = np.arange(1, len(result_same)+1)
print(result_same.head(5))

########################333#OD

od_matrix = df.groupby(['LocationID_x', 'LocationID_y']).size().unstack(fill_value=0)
#print(od_matrix)
# Get the top 25 zones based on total trips
top_zones = od_matrix.sum(axis=1).nlargest(25).index
empty_vendor_trips=empty_vendor_trips.loc[top_zones,top_zones]

filtered_od_matrix = od_matrix.loc[top_zones, top_zones]
filtered_od_matrix=empty_vendor_trips+ filtered_od_matrix
#print(filtered_od_matrix)

filtered_od_matrix.loc['Total'] = filtered_od_matrix.sum()  # Sum for each drop-off column
filtered_od_matrix['Total'] = filtered_od_matrix.sum(axis=1)

#print(filtered_od_matrix.info)
#print(filtered_od_matrix['Total'])
#print(filtered_od_matrix.T['Total'])
if filtered_od_matrix['Total'].sum() == filtered_od_matrix.T['Total'].sum():
    print("The OD matrix is balanced.")
else:
    print("The OD matrix is not balanced.")
# Plotting the heatmap for the filtered OD matrix
plt.figure(figsize=(12, 10))
sns.heatmap(filtered_od_matrix, cmap="YlGnBu", annot=True, fmt=".0f", cbar_kws={"label": "Number of Trips"},annot_kws={"size": 5})
plt.title("Top 25 Origin-Destination Matrix Heatmap")
plt.xlabel("Drop-off Zone")
plt.ylabel("Pickup Zone")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

##هزینه کلی
# ساخت ماتریس خالی برای میانگین هزینه‌ها
cost_matrix = pd.DataFrame(index=top_zones, columns=top_zones).fillna(0)

# پر کردن ماتریس با میانگین هزینه برای هر جفت مبدا و مقصد
for pickup_zone in top_zones:
    for dropoff_zone in top_zones:
        # فیلتر کردن داده‌ها برای زون‌های مبدا و مقصد خاص
        filtered_data = df[(df['LocationID_x'] == pickup_zone) & (df['LocationID_y'] == dropoff_zone)]
        
        # محاسبه میانگین هزینه
        if not filtered_data.empty:
            avg_cost = filtered_data['total_amount'].mean()
        else:
            avg_cost = 0  # اگر داده‌ای وجود نداشت، مقدار صفر قرار داده می‌شود
        
        # پر کردن ماتریس با مقدار میانگین هزینه
        cost_matrix.loc[pickup_zone, dropoff_zone] = avg_cost

# نمایش ماتریس میانگین هزینه
print(cost_matrix)

# # رسم نمودار Heatmap برای ماتریس میانگین هزینه
# plt.figure(figsize=(12, 10))
# sns.heatmap(cost_matrix.astype(float), cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={"label": "Average Cost"}, annot_kws={"size": 7})
# plt.title("Average Cost Matrix for Top Zones")
# plt.xlabel("Drop-off Zone")
# plt.ylabel("Pickup Zone")
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.show()
