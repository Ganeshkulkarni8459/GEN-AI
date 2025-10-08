import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')
df = df.drop(["CustomerID","Gender","Age"],axis=1)
df.rename(columns={"Annual Income (k$)":"Income","Spending Score (1-100)":"Spending"},inplace=True)

wss = []
cluster_range = range(1,11)

for k in cluster_range:
    model = KMeans(n_clusters=k, random_state=123)
    model.fit(df)
    wss.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(cluster_range, wss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WSS)')
plt.grid(True)
plt.show()