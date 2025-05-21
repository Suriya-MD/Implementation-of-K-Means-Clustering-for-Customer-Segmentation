# Exp-10 Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start by importing the required libraries (pandas, matplotlib.pyplot, KMeans from sklearn.cluster).

2.Load the Mall_Customers.csv dataset into a DataFrame.

3.Check for missing values in the dataset to ensure data quality.

4.Select the features Annual Income (k$) and Spending Score (1-100) for clustering.

5.Use the Elbow Method by running KMeans for cluster counts from 1 to 10 and record the Within-Cluster Sum of Squares (WCSS).

6.Plot the WCSS values against the number of clusters to determine the optimal number of clusters (elbow point).

7.Fit the KMeans model to the selected features using the chosen number of clusters (e.g.,5).

8.Predict the cluster label for each data point and assign it to a new column called cluster.

9.Split the dataset into separate clusters based on the predicted labels.

10.Visualize the clusters using a scatter plot, and optionally mark the cluster centroids. 

## Program:
```

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SURIYA M
Register Number:  212223110055

```

```
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")


data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = [] #within cluster sum of square.
#It is the sum of square distance between each point & the centroid in cluster

for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11),wcss)
plt.xlabel("No.of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

y_pred = km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred

df0=data[data["cluster"]==0]

df1=data[data["cluster"]==1]

df2=data[data["cluster"]==2]

df3=data[data["cluster"]==3]

df4=data[data["cluster"]==4]

plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")

plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")

plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")

plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")

plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")

plt.legend()

plt.title("Customer Segment")
```














## Output:

![image](https://github.com/user-attachments/assets/351ea76a-29bc-4ef3-83e3-b4ab4488d312)


![image](https://github.com/user-attachments/assets/075f9c8e-a194-42f9-96ab-b56a2b407012)


![image](https://github.com/user-attachments/assets/94382957-ff8e-4ec6-a8be-87d540d77fa7)


![image](https://github.com/user-attachments/assets/d3494e72-5e2c-413f-b1a5-9bbd5ef81e55)


![image](https://github.com/user-attachments/assets/7fa64551-a5c9-4ac3-ad85-c029232b3520)


![image](https://github.com/user-attachments/assets/994f9c15-fc21-45cf-b131-3b26cecb797e)

![image](https://github.com/user-attachments/assets/b8ad769d-45fb-4ad0-b28b-1d8b1c7bf8ea)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
