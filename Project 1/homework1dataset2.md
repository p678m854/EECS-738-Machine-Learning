# Roses are Red
This project is the the second data set where I used my kmeans algorithm to cluster data and find centroids of data. We picked the iris data set from kaggle to use the algorithm on. I picked the y-axis to be petal length and the x-axis to be petal width. These are measurements of the flower. I used different colors to cluster the different species of iris. Go into terminal and run command python3 flower_clusters.py You will see four different boxes pop up. The top two boxes are the data points plotted, one with robot moves in color and the other with non-colored data. This gives you an opportunity to see the data points plotted. The bottom two graphs show the kmeans algorithm vs the actual known iris species.

# How the K Means Algorithm works:
So first we initialize the kmeans class with our training data and the number of clusters we want. Then we randomize the number of data points and use the first number of points to randomly place our centroids. We then iterate through every data point and calculate the distance for every centroid. we then pick the minimum distance and determine the nearest centroid and store that into an array. We then go through the associated data points to each centroid and calculate the total mean for those data points, then we are able to place the centroid based on that calculation. We then interate through this process a certain amount of times to achieve the minimum distance for our clusters and their associated data points.

# Conclusion
This is fairly good correlation with the different species types and leaf dimensions. 
