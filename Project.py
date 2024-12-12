import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
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
df.loc[df['passenger_count'] > 4, 'payment_type'] = 6
df.loc[df['passenger_count'] > 4, 'trip_type'] = 3
df.dropna(subset=['VendorID'], inplace=True)

# Convert pickup datetime
df['pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df = df[df['pickup_datetime'].dt.year == 2021]
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
print(missing_value_df)
# Extract hour from pickup_datetime
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_day'] = df['pickup_datetime'].dt.day
df['pickup_month'] = df['pickup_datetime'].dt.month

# Extract dropoff datetime
df['dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df['dropoff_hour'] = df['dropoff_datetime'].dt.hour
df['dropoff_day'] = df['dropoff_datetime'].dt.day
df['dropoff_month'] = df['dropoff_datetime'].dt.month

# Drop original datetime columns
df.drop(columns=['lpep_pickup_datetime', 'lpep_dropoff_datetime'], inplace=True)

# Merge borough information into the main DataFrame
borough_df = pd.read_csv('Boroughs.csv')
df = df.merge(borough_df[['LocationID', 'Borough']], left_on='PULocationID', right_on='LocationID', how='left')
df.rename(columns={'Borough': 'PU_Borough'}, inplace=True)

df = df.merge(borough_df[['LocationID', 'Borough']], left_on='DOLocationID', right_on='LocationID', how='left')
df.rename(columns={'Borough': 'DO_Borough'}, inplace=True)

# One-hot encode boroughs
df = pd.get_dummies(df, columns=['PU_Borough'], prefix=['PU'], drop_first=True)
df.drop([ 'PULocationID', 'DOLocationID', 'LocationID_x', 'LocationID_y','pickup_datetime', 'dropoff_datetime' ,'DO_Borough'], axis=1, inplace=True)
print(df.columns)
df = df.apply(pd.to_numeric, errors='coerce')
# Handle missing value using regression or classification
for column in df.columns:
    null_percentage = df[column].isnull().mean()
    if 0.05 < null_percentage <= 0.3:
        missing_rows = df[df[column].isnull()]
        complete_rows = df[df[column].notnull()]

        # Ensure you drop datetime columns prior to fitting
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

print(df.head())
print(df.info())

# Calculate and display missing values percentage
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing': percent_missing})
print(missing_value_df)

# Extract hour from pickup_datetime
df_1=df[:300000]
# 1. Trip Type Distribution by Hour  
trip_type_hourly_distribution = df_1.groupby(['pickup_hour', 'trip_type']).size().unstack(fill_value=0)

# Plotting the Trip Type Distribution by Hour  
plt.figure(figsize=(14, 7))  
sns.barplot(data=trip_type_hourly_distribution.reset_index().melt(id_vars=['pickup_hour']),
            x='pickup_hour', y='value', hue='trip_type', errorbar=None)
plt.title('Trip Type Distribution by Hour of Day')  
plt.xlabel('Hour of Day')  
plt.ylabel('Number of Trips')  
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability  
plt.legend(title='Trip Type', bbox_to_anchor=(1.05, 1), loc='upper left')  
plt.tight_layout()  
plt.show()  

# 2. Payment Type Distribution by Hour  
payment_type_hourly_distribution = df_1.groupby(['pickup_hour', 'payment_type']).size().unstack(fill_value=0)

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
taxi_df = df_1[df_1['trip_type'] != 3]
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
# Grouping by all dummy variables for boroughs and pickup hour
green_taxi_trend = df_1.groupby(['pickup_hour'] + [col for col in df.columns if col.startswith('PU_') or col.startswith('DO_')]).size().unstack(fill_value=0)

# Plotting the trend of green taxi usage by boroughs
plt.figure(figsize=(14, 7))
green_taxi_trend.T.plot(kind='bar', figsize=(14, 7))
plt.title('Green Taxi Usage Trend by Borough')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Trips')
plt.legend(title='Borough', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()


##############            بخش دوم
# محاسبه ماتریس همبستگی
correlation_matrix = df_1.corr()
# رسم ماتریس همبستگی به صورت Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix for Predicting Total Amount')
plt.show()

# تحلیل: ستون‌هایی که بیشترین همبستگی را با total_amount دارند
correlation_with_total = correlation_matrix['total_amount'].sort_values(ascending=False)
print("Top correlations with total_amount:")
print(correlation_with_total)

##### بخش دوم خواسته سوم

def backward_feature_selection(data, target_column='total_amount', n_features=5):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return list(selected_features)

backward_selected_features = backward_feature_selection(df_1, target_column='total_amount', n_features=5)
print("Selected Features:", backward_selected_features)


##### بخش دوم خواسته سوم

def forward_feature_selection(data, target_column='total_amount', n_features=5):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    model = LinearRegression()
    selected_features = []
    remaining_features = list(X.columns)
    while len(selected_features) < n_features and remaining_features:
        best_score = -np.inf
        best_feature = None
        for feature in remaining_features:
            trial_features = selected_features + [feature]
            scores = cross_val_score(model, X[trial_features], y, cv=5, scoring='r2')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_feature = feature
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
    return selected_features
forward_selected_features = forward_feature_selection(df_1, target_column='total_amount', n_features=5)
print("Selected features:", forward_selected_features)


def random_forest_feature_selection(data, target_column='total_amount', n_features=5):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)

    selected_features = feature_importances.nlargest(n_features).index.tolist()
    return selected_features
rf_selected_features = random_forest_feature_selection(df_1, target_column='total_amount', n_features=5)
print("Selected Features:", rf_selected_features)