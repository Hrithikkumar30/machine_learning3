# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 15:00:26 2021

@author: alfa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.cluster import KMeans

customer_data = pd.read_csv("C:/Users/alfa/Desktop/machineLearning/coustomer_segmentation/Mall_Customers.csv")
print(customer_data.head)

#print(customer_data.info())

x = customer_data.iloc[:,[3,4]].values  #Choosing the anuual income column and spending score column
#print(x)

# finding WSS value for different number of cluster Within cluster sum of square

wss=[]
for i in range (1,11):
    kmeans = KMeans (n_clusters=i , init="k-means++" , random_state= 24)
    kmeans.fit(x)
    wss.append(kmeans.inertia_)
    
    #PLOTTING ELBOW GRAPH

sbn.set()
plt.plot(range(1,11),wss)
plt.title("the elbow graph")
plt.xlabel("no of cluster")
plt.ylabel("value of wss")
plt.show()

 #training the kmeans model
kmeans = KMeans (n_clusters=5 , init="k-means++" , random_state= 2)
Y = kmeans.fit_predict(x)
#print(Y)
 
#visualizing the cluster 

plt.figure(figsize=(8,8))
plt.scatter(x[Y==0,0], x[Y==0,1], s=50, c="green" , label="CLuster1")
plt.scatter(x[Y==1,0], x[Y==1,1], s=50, c="blue" , label="CLuster2")
plt.scatter(x[Y==2,0], x[Y==2,1], s=50, c="red" , label="CLuster3")
plt.scatter(x[Y==3,0], x[Y==3,1], s=50, c="orange" , label="CLuster4")
plt.scatter(x[Y==4,0], x[Y==4,1], s=50, c="yellow" , label="CLuster5")


plt.scatter(kmeans.cluster_centers_[:0],kmeans.cluster_centers_[:1] , s=100, c="blackk" , label ="centroid")