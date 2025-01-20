import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
data = pd.read_csv("healthcare_dataset.csv")

# Step 2: Preprocess the dataset
# Drop non-numeric columns or those irrelevant to clustering
numeric_data = data.select_dtypes(include=[np.number]).dropna()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Step 3: Determine the optimal number of clusters using the Elbow method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

# Step 4: Fit the K-Means model with the optimal number of clusters
optimal_k = 3  # Assume we determine 3 clusters from the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 5: Analyze clusters
# Add cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Summary of clusters
#print(data.groupby('Cluster').mean())

# Visualize the clusters (if data is 2D or reducible)
sns.scatterplot(
    x=scaled_data[:, 0], y=scaled_data[:, 1], hue=data['Cluster'], palette='viridis'
)
plt.title("Cluster Visualization")
plt.show()