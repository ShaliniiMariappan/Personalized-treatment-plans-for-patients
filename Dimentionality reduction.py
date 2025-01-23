import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

file_path = "healthcare_dataset.csv"
data = pd.read_csv(file_path)

print(data.head())

numeric_data = data.select_dtypes(include=[np.number]).dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

pca = PCA()
pca_data = pca.fit_transform(scaled_data)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components to retain 95% variance: {n_components}")

pca_reduced = PCA(n_components=n_components)
reduced_data = pca_reduced.fit_transform(scaled_data)

columns = [f"PC{i+1}" for i in range(n_components)]
reduced_df = pd.DataFrame(reduced_data, columns=columns)

print(reduced_df.head())
