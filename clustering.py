# In this notebook we practice k-means clustering with 2 examples:

# -   k-means on a random generated dataset
# -   Using k-means for customer segmentation
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
#%matplotlib inline

import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 

from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 

from sklearn.cluster import DBSCAN 

from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 

#Lets create our own dataset for this lab!

def k_means():
    #Lets create our own dataset for this lab!
    np.random.seed(0)
    X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
    plt.scatter(X[:, 0], X[:, 1], marker='.')
    #plt.show()

    #The KMeans class has many parameters that can be used, but we will be using these three
    #method of the centroids" k-means++
    #The number of clusters to form as well as the number of centroids to generate: 4
    #Number of time the k-means algorithm will be run with different centroid seeds.: 12. final result will be th best of the 12
    k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
    #Now let's fit the KMeans model with the feature matrix we created above
    k_means.fit(X)
    #Now let's grab the labels for each point in the model using KMeans' <b> .labels_ </b> attribute and save it as <b> k_means_labels </b>
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_

    # Initialize the plot with the specified dimensions.
    fig = plt.figure(figsize=(6, 4))

    # Colors uses a color map, which will produce an array of colors based on
    # the number of labels there are. We use set(k_means_labels) to get the
    # unique labels.
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

    # Create a plot
    ax = fig.add_subplot(1, 1, 1)

    # For loop that plots the data points and centroids.
    # k will range from 0-3, which will match the possible clusters that each
    # data point is in.
    for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

        # Create a list of all data points, where the data poitns that are 
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        my_members = (k_means_labels == k)

        # Define the centroid, or cluster center.
        cluster_center = k_means_cluster_centers[k]

        # Plots the datapoints with color col.
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')

        # Plots the centroids with specified color, but with a darker outline
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

    # Title of the plot
    ax.set_title('KMeans')

    # Remove x-axis ticks
    ax.set_xticks(())

    # Remove y-axis ticks
    ax.set_yticks(())

    # Show the plot
    plt.show()

def hieClust():
    #     We will be generating a set of data using the <b>make_blobs</b> class. <br> <br>
    # Input these parameters into make_blobs:
    # <ul>
    #     <li> <b>n_samples</b>: The total number of points equally divided among clusters. </li>
    #     <ul> <li> Choose a number from 10-1500 </li> </ul>
    #     <li> <b>centers</b>: The number of centers to generate, or the fixed center locations. </li>
    #     <ul> <li> Choose arrays of x,y coordinates for generating the centers. Have 1-10 centers (ex. centers=[[1,1], [2,5]]) </li> </ul>
    #     <li> <b>cluster_std</b>: The standard deviation of the clusters. The larger the number, the further apart the clusters</li>
    #     <ul> <li> Choose a number between 0.5-1.5 </li> </ul>
    # </ul> <br>
    # Save the result to <b>X1</b> and <b>y1</b>.
    X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o') 

    agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
    agglom.fit(X1,y1)

    # Create a figure of size 6 inches by 4 inches.
    plt.figure(figsize=(6,4))

    # These two lines of code are used to scale the data points down,
    # Or else the data points will be scattered very far apart.

    # Create a minimum and maximum range of X1.
    x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

    # Get the average distance for X1.
    X1 = (X1 - x_min) / (x_max - x_min)

    # This loop displays all of the datapoints.
    for i in range(X1.shape[0]):
        # Replace the data points with their respective cluster value 
        # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
        plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
                color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
        
    # Remove the x ticks, y ticks, x and y axis
    plt.xticks([])
    plt.yticks([])
    #plt.axis('off')

    # Display the plot of the original data before clustering
    plt.scatter(X1[:, 0], X1[:, 1], marker='.')
    # Display the plot
    plt.show()

    dist_matrix = distance_matrix(X1,X1) 
    print(dist_matrix)

    Z = hierarchy.linkage(dist_matrix, 'complete')

    # A Hierarchical clustering is typically visualized as a dendrogram as shown in the following cell. Each merge is represented by a horizontal line. The y-coordinate of the horizontal line is the similarity of the two clusters that were merged, where cities are viewed as singleton clusters. 
    # By moving up from the bottom layer to the top node, a dendrogram allows us to reconstruct the history of merges that resulted in the depicted clustering. 

    # Next, we will save the dendrogram to a variable called <b>dendro</b>. In doing this, the dendrogram will also be displayed.
    # Using the <b> dendrogram </b> class from hierarchy, pass in the parameter:
    dendro = hierarchy.dendrogram(Z)
    print(dendro.values())

def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    # Create random data and store in feature matrix X and response vector y.
    X, y = make_blobs(n_samples=numSamples, centers=centroidLocation, 
                                cluster_std=clusterDeviation)
    
    # Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)
    return X, y

def dbscan():#DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise
    X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)

    # It works based on two parameters: Epsilon and Minimum Points  
    # **Epsilon** determine a specified radius that if includes enough number of points within, we call it dense area  
    # **minimumSamples** determine the minimum number of data points we want in a neighborhood to define a cluster.
    epsilon = 0.3
    minimumSamples = 7
    db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
    labels = db.labels_

    #Lets Replace all elements with 'True' in core_samples_mask that are in the cluster, 'False' if the points are outliers.
    # Firts, create an array of booleans using the labels from db.
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # Remove repetition in labels by turning it into a set.
    unique_labels = set(labels)

    # Create colors for the clusters.
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # Plot the points with colors
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        # Plot the datapoints that are clustered
        xy = X[class_member_mask & core_samples_mask]
        plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

        # Plot the outliers
        xy = X[class_member_mask & ~core_samples_mask]
        plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

    plt.show()
    

#density based clustering
dbscan()
#Agglomerative Hierarchical Clustering
#hieClust()
#k_means()

