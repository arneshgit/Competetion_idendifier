# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:14:54 2024

@author: Dell
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('C:/Users/Dell/.p2/Downloads/output_bottom.csv')

# Handle missing values if any
df.fillna(df.mean(), inplace=True)

# Convert date columns to datetime
df['startDate'] = pd.to_datetime(df['startDate'], errors='coerce')
df['endDate'] = pd.to_datetime(df['endDate'], errors='coerce')

# Drop rows with invalid dates
df.dropna(subset=['startDate', 'endDate'], inplace=True)

# Feature engineering
df['hour'] = df['startDate'].dt.hour
df['day_of_week'] = df['startDate'].dt.dayofweek

# Generate labels for faults and resource utilization
fault_threshold = df['sPackets'].quantile(0.95)
df['fault'] = (df['sPackets'] > fault_threshold).astype(int)
df['resource_utilization'] = df['sBytesSum'] / df['rBytesSum']

# Select features
features = ['sPackets', 'rPackets', 'sBytesSum', 'rBytesSum', 'hour', 'day_of_week']
X = df[features]
y_fault = df['fault']
y_resource = df['resource_utilization']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ensure no NaN or infinite values
if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
    raise ValueError("Data contains NaN or infinite values after filling.")


import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set the index to the startDate
df.set_index('startDate', inplace=True)

# Resample data by hour and compute the sum of packets
hourly_traffic = df['sPackets'].resample('H').sum()

# Fill missing values in resampled data
hourly_traffic.fillna(0, inplace=True)

# Check the shape of the resampled data
print("Shape of hourly traffic data:", hourly_traffic.shape)
print(hourly_traffic.head(24))  # Display the first 24 entries

# Ensure there are enough observations
seasonal_periods = 24 if hourly_traffic.shape[0] >= 48 else 12

# Train a Holt-Winters model
model = ExponentialSmoothing(hourly_traffic, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
fit = model.fit()

# Predict future traffic
future_steps = 24
future_traffic = fit.forecast(steps=future_steps)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(hourly_traffic.index, hourly_traffic, label='Observed')
plt.plot(future_traffic.index, future_traffic, label='Forecast', linestyle='--')
plt.title('Hourly Network Traffic Prediction')
plt.xlabel('Time')
plt.ylabel('Number of Packets')
plt.legend()
plt.show()


from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Anomaly detection using Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(X_scaled)

# Predict anomalies
df['anomaly'] = clf.predict(X_scaled)
df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Filter out the anomalies
anomalies = df[df['anomaly'] == 1]
print("Anomalies detected:")
print(anomalies)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('C:/Users/Dell/.p2/Downloads/output_bottom.csv')

# Handle missing values
df.fillna(df.mean(), inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Convert date columns to datetime
df['startDate'] = pd.to_datetime(df['startDate'], errors='coerce')
df.set_index('startDate', inplace=True)

# Define performance features
performance_features = ['sPackets', 'rPackets', 'sBytesSum', 'rBytesSum']
target_feature = 'sLoad'  # Assuming 'sLoad' is your target variable

# Anomaly Detection using Isolation Forest
X = df[performance_features]
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(X)
df['anomaly'] = clf.predict(X)
df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Filter out the anomalies
anomalies = df[df['anomaly'] == 1]
print(anomalies)

# AI-Based Prediction using Gradient Boosting
y = df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Add predictions to the main DataFrame
df['predicted_load'] = model.predict(X)

# Final DataFrame with anomalies and predictions
final_df = df[['predicted_load', 'anomaly']]
print(final_df.head())

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['predicted_load'], label='Predicted Load', color='orange')
plt.scatter(df[df['anomaly'] == 1].index, df['predicted_load'][df['anomaly'] == 1], color='red', label='Anomalies')
plt.title('Predicted Resource Utilization with Anomalies')
plt.xlabel('Time')
plt.ylabel('Predicted Load')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load and preprocess your DataFrame
df = pd.read_csv('C:/Users/Dell/.p2/Downloads/output_bottom.csv')
df.fillna(df.mean(), inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Define performance features
performance_features = ['sPackets', 'rPackets', 'sBytesSum', 'rBytesSum']

# Add attack label
attack_threshold = df['sBytesSum'].quantile(0.95)
df['attack'] = (df['sBytesSum'] > attack_threshold).astype(int)

# Prepare features and target
X = df[performance_features]
y = df['attack']

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for classification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train the classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Group by source and destination IPs and summarize flow statistics
flow_summary = df.groupby(['sIPs', 'rIPs']).agg({
    'sPackets': 'sum',
    'rPackets': 'sum',
    'sBytesSum': 'sum',
    'rBytesSum': 'sum'
}).reset_index()

# Sort by the number of packets sent
flow_summary = flow_summary.sort_values(by='sPackets', ascending=False)

# Display the top 10 flows
print(flow_summary.head(10))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Load your data
df = pd.read_csv('C:/Users/Dell/.p2/Downloads/output_bottom.csv')
df.fillna(df.mean(), inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Define performance features and scale them
performance_features = ['sPackets', 'rPackets', 'sBytesSum', 'rBytesSum']
X = df[performance_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['load_cluster'] = kmeans.fit_predict(X_scaled)

# Anomaly detection using Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = clf.fit_predict(X_scaled)
df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Visualization of load clusters
sns.scatterplot(data=df, x='sBytesSum', y='rBytesSum', hue='load_cluster')
plt.title('Load Distribution Clusters')
plt.xlabel('Sent Bytes Sum')
plt.ylabel('Received Bytes Sum')
plt.show()

# Visualization of anomalies
sns.scatterplot(data=df, x='sPackets', y='rPackets', hue='anomaly')
plt.title('Anomalies in Network Traffic')
plt.xlabel('Sent Packets')
plt.ylabel('Received Packets')
plt.show()

# Example visualization of top flows
flow_summary = df.groupby(['sIPs', 'rIPs']).agg({
    'sPackets': 'sum',
    'rPackets': 'sum',
    'sBytesSum': 'sum',
    'rBytesSum': 'sum'
}).reset_index()

top_flows = flow_summary.head(10)
sns.barplot(data=top_flows, x='sIPs', y='sPackets')
plt.title('Top 10 Network Flows by Sent Packets')
plt.xlabel('Source IPs')
plt.ylabel('Sent Packets')
plt.xticks(rotation=45)
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv('C:/Users/Dell/.p2/Downloads/output_bottom.csv')

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Feature selection
features = ['sPackets', 'rPackets', 'sBytesSum', 'rBytesSum', 'sPayloadSum', 'rPayloadSum']
X = df[features]

# Define y for anomaly detection
anomaly_threshold = df['sPackets'].quantile(0.95)
df['anomaly'] = (df['sPackets'] > anomaly_threshold).astype(int)
y_anomaly = df['anomaly']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for anomaly detection
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_anomaly, test_size=0.3, random_state=42)

# Train a Random Forest for anomaly detection
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate anomaly detection
accuracy_anomaly = accuracy_score(y_test, y_pred)
print(f"Anomaly Detection Accuracy: {accuracy_anomaly}")

# For attack prediction, similarly define y_attack
attack_threshold = df['sBytesSum'].quantile(0.95)
df['attack'] = (df['sBytesSum'] > attack_threshold).astype(int)
y_attack = df['attack']

# Split data for attack prediction
X_train_attack, X_test_attack, y_train_attack, y_test_attack = train_test_split(X_scaled, y_attack, test_size=0.3, random_state=42)

# Train a Random Forest for attack prediction
clf_attack = RandomForestClassifier(random_state=42)
clf_attack.fit(X_train_attack, y_train_attack)
y_pred_attack = clf_attack.predict(X_test_attack)

# Evaluate attack prediction
accuracy_attack = accuracy_score(y_test_attack, y_pred_attack)
print(f"Attack Prediction Accuracy: {accuracy_attack}")

# For load distribution clustering, use KMeans or similar methods...

# For linear regression, if applicable
# Define y_resource and split as needed


from sklearn.metrics import confusion_matrix, classification_report

# Evaluate anomaly detection
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{report}")

# For attack prediction
confusion_attack = confusion_matrix(y_test_attack, y_pred_attack)
report_attack = classification_report(y_test_attack, y_pred_attack)

print(f"Attack Confusion Matrix:\n{confusion_attack}")
print(f"Attack Classification Report:\n{report_attack}")


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Make predictions for the anomaly detection model
y_pred_anomaly = clf.predict(X_test)

# Evaluate the model
print("Anomaly Detection Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_anomaly)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_anomaly))
print("Classification Report:")
print(classification_report(y_test, y_pred_anomaly))


# Split data for classification (adjust X and y as needed)
X_train_attack, X_test_attack, y_train_attack, y_test_attack = train_test_split(X_scaled, df['attack'], test_size=0.3, random_state=42)

# Train a classifier (e.g., Random Forest, SVM)
clf_attack = RandomForestClassifier(random_state=42)
clf_attack.fit(X_train_attack, y_train_attack)

# Make predictions for the attack prediction model
y_pred_attack = clf_attack.predict(X_test_attack)

# Evaluate the model
print("Attack Prediction Evaluation:")
print(f"Accuracy: {accuracy_score(y_test_attack, y_pred_attack)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_attack, y_pred_attack))
print("Classification Report:")
print(classification_report(y_test_attack, y_pred_attack))

from sklearn.model_selection import cross_val_score

# For anomaly detection model
clf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(clf, X_scaled, y_anomaly, cv=5)  # 5-fold cross-validation
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {cv_scores.mean()}")


import seaborn as sns
import matplotlib.pyplot as plt

# Check the distribution of the target variable for anomalies
sns.countplot(x='anomaly', data=df)
plt.title('Distribution of Anomaly Labels')
plt.show()

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example logging in your DataFrame operations
logging.info(f"Original DataFrame shape: {df.shape}")
df.fillna(df.mean(), inplace=True)
logging.info(f"DataFrame shape after filling NaNs: {df.shape}")

# Log after defining features
logging.info(f"Features selected: {performance_features}")
X = df[performance_features]
logging.info(f"Features shape: {X.shape}")

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f'Attack Prediction Accuracy: {accuracy}')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assuming y_test and y_pred are the true and predicted labels
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Attack', 'Attack'])
disp.plot()
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Load the data
df = pd.read_csv('C:/Users/Dell/.p2/Downloads/output_bottom.csv')


# Ensure 'anomaly' column exists
if 'anomaly' not in df.columns:
    df['anomaly'] = 0  # Default to 0 if not present

# Initialize the app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children='Network Management Dashboard'),

    # Summary statistics
    html.Div(children=[
        html.H2('Summary Statistics'),
        html.Div(children=[
            html.Div(children=[
                html.H3('Total Packets Sent'),
                html.P(df['sPackets'].sum())
            ], className='stat-card'),
            html.Div(children=[
                html.H3('Total Packets Received'),
                html.P(df['rPackets'].sum())
            ], className='stat-card'),
            html.Div(children=[
                html.H3('Average Payload Size'),
                html.P(df['sPayloadAvg'].mean())
            ], className='stat-card'),
        ], className='stat-container'),
    ]),

    # Correlation matrix heatmap
    html.Div(children=[
        html.H2('Correlation Matrix'),
        dcc.Graph(
            id='correlation-heatmap',
            figure=px.imshow(df[['sPackets', 'rPackets', 'sBytesSum', 'rBytesSum', 'sPayloadSum', 'rPayloadSum']].corr(), 
                             color_continuous_scale='RdBu_r', 
                             title='Correlation Matrix')
        )
    ]),

    # Network traffic analysis
    html.Div(children=[
        html.H2('Top 10 Network Flows'),
        dcc.Graph(
            id='top-flows',
            figure=px.bar(df.groupby(['sIPs', 'rIPs']).agg({'sPackets': 'sum'}).reset_index().sort_values(by='sPackets', ascending=False).head(10), 
                          x='sIPs', y='sPackets', color='rIPs', title='Top 10 Network Flows')
        )
    ]),

    # Anomaly detection
    html.Div(children=[
        html.H2('Anomalies Detected'),
        dcc.Graph(
            id='anomalies',
            figure=px.scatter(df[df['anomaly'] == 1], x='sPackets', y='rPackets', color='anomaly', 
                              title='Anomalies Detected',
                              labels={'anomaly': 'Anomaly Detected'})
        )
    ]),

    # Protocol usage
    html.Div(children=[
        html.H2('Protocol Usage'),
        dcc.Graph(
            id='protocol-usage',
            figure=px.bar(df['protocol'].value_counts().reset_index(), x='index', y='protocol', 
                          title='Protocol Usage',
                          labels={'index': 'Protocol', 'protocol': 'Count'})
        )
    ]),

    # Latency analysis
    html.Div(children=[
        html.H2('Latency Analysis'),
        dcc.Graph(
            id='latency',
            figure=go.Figure(data=[
                go.Scatter(x=df['startDate'], y=df['sAckDelayAvg'], mode='lines', name='sAckDelayAvg'),
                go.Scatter(x=df['startDate'], y=df['rAckDelayAvg'], mode='lines', name='rAckDelayAvg')
            ]).update_layout(title='ACK Delay Averages', xaxis_title='Date', yaxis_title='ACK Delay')
        )
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
