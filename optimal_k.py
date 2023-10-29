from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_scores = [] # holds the silhouette scores for each k
K = range(2, 10) # change accordingly

for k in K:
    kmeans = KMeans(n_clusters=k).fit(images)
    label = kmeans.labels_
    sil_coeff = silhouette_score(images, label, metric='euclidean')
    silhouette_scores.append(sil_coeff)

# The optimal number of clusters is the one that maximizes the silhouette score
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]

print(f'The optimal number of clusters is: {optimal_k}')