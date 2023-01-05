import dataset as dataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
dataset=pd.read_csv('data.csv')
x=dataset.iloc[[1,2]].values
print(x)
from sklearn.cluster import KMeans
wcss_list=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',test_size=0.3,random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)

y_predict = kmeans.fit(x)

kmeans=KMeans(n_clusters=3,init='k-means++',test_size=0.3,random_state=42)
plt.scatter(x[y_predict==0,0],x[y_predict==1,0],s=100,c='blue',label='cluster1')
plt.scatter(x[y_predict==1,0],x[y_predict==1,1],s=100,c='red',label='cluster2')
plt.scatter(x[y_predict==2,0],x[y_predict==2,1],s=100,c='green',label='cluster3')
mtp.plot('The Elbow method graph')
mtp.xlabel('Annual Income')
mtp.ylabel('Spending score')
