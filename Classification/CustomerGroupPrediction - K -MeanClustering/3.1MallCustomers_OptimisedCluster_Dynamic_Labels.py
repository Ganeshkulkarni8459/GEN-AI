import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv')
df = df.drop(["CustomerID","Gender","Age"],,axis=1)
df.rename(columns={"Annual Income (k$)":"Income","Spending Score (1-100)":"Spending"},inplace=True)

kmeans_final = KMeans(n_clusters=5, random_state=123)
df['Cluster'] = kmeans_final.fit_predict(df)

centroids = kmeans_final.cluster_centers_
print("Cluster Center: ")
print(centroids)

cluster_labels = {}
for cid , (income,spending) in enumerate(centroids):
    if income < 40 and spending < 40:
        cluster_labels[cid] = 'Budget-Conscious'
    elif income < 40 and spending >= 60:
        cluster_labels[cid] = 'Impulsive Spenders'
    elif income >= 70 and spending < 40:
        cluster_labels[cid] = 'Savers'
    elif income >= 70 and spending >= 60:
        cluster_labels[cid] = 'Premium Spenders'
    else:
        cluster_labels[cid] = 'Average Customers'


df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

print("Cluster Labels Assigned: ")
print(df.head(10))

