import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv')
df = df.drop(["CustomerID","Gender","Age"],axis=1)
df.rename(columns={"Annual Income (k$)":"Income","Spending Score (1-100)":"Spending"},inplace=True)

kmeans_final = KMeans(n_clusters=5, random_state=123)
df['Cluster'] = kmeans_final.fit_predict(df)

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Income', y='Spending', hue='Cluster', palette='Set1', s=100)
plt.title('Customer Segments based on Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()