# Spotify Genre Grouping Project
# Complete, clean, and submission-ready code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv('spotify dataset.csv')

# ----------------------------
# 2. Data Preprocessing
# ----------------------------

# Select numerical audio features
features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo'
]

# Handle missing values
for col in features:
    df[col].fillna(df[col].mean(), inplace=True)

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# ----------------------------
# 3. Exploratory Data Analysis
# ----------------------------

# Distribution plots
for col in ['danceability', 'energy', 'valence']:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Scatter plot
plt.figure()
sns.scatterplot(x='energy', y='danceability', data=df)
plt.title('Energy vs Danceability')
plt.show()

# Boxplot (genre comparison)
plt.figure(figsize=(10, 5))
sns.boxplot(x='playlist_genre', y='loudness', data=df)
plt.xticks(rotation=45)
plt.title('Loudness across Playlist Genres')
plt.show()

# ----------------------------
# 4. Correlation Matrix
# ----------------------------

plt.figure(figsize=(10, 8))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Audio Features')
plt.show()

# ----------------------------
# 5. Elbow Method
# ----------------------------

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# ----------------------------
# 6. K-Means Clustering
# ----------------------------

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# ----------------------------
# 7. PCA Visualization
# ----------------------------

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

plt.figure()
plt.scatter(
    pca_features[:, 0],
    pca_features[:, 1],
    c=df['Cluster']
)
plt.title('PCA Visualization of Song Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# ----------------------------
# 8. Cluster vs Genre Analysis
# ----------------------------

cluster_genre = pd.crosstab(df['Cluster'], df['playlist_genre'])
print("\nCluster vs Playlist Genre Distribution:\n")
print(cluster_genre)

# ----------------------------
# 9. Recommendation Example
# ----------------------------

def recommend_songs(song_index, n_recommendations=5):
    cluster = df.loc[song_index, 'Cluster']
    recommendations = df[df['Cluster'] == cluster].sample(n_recommendations)
    return recommendations[['track_name', 'playlist_genre']]

# Example usage
print("\nRecommended Songs:\n")
print(recommend_songs(0))
