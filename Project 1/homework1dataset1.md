# 1976 Vehicle Data and K Means
This project is the the first data set where I used my kmeans algorithm to cluster data and find centroids of data.
We picked the auto-mpg data set from kaggle to use my algorithm on.
We picked the y-axis to be mpg and the x-axis to be weight of vehicles.  I used different colors to cluster the amount of cylinders each car had.  The vehicle data is from the year 1976.
Go into terminal and run command python3 cluster_cylinders.py
You will see four different boxes pop up.  The top two boxes are the data points plotted,  one with cylinders in color and the other with non-colored data.  This gives you an opportunity to see the data points plotted.
The bottom two graphs show the kmeans algorithm vs the actual known cylinders.  

# How the kmeans algorithm works:
So first we initialize the kmeans class with our training data and the number of clusters we want.
Then we randomize the number of data points and use the first number of points to randomly place our centroids.
We then iterate through every data point and calculate the distance for every centroid.  we then pick the minimum distance and determine the nearest centroid and store that into an array.
We then go through the associated data points to each centroid and calculate the total mean for those data points,  then we are able to place the centroid based on that calculation.
We then interate through this process a certain amount of times to achieve the minimum distance for our clusters and their associated data points.

# How the data was organized:
We used vehicle dataset,  we then used pandas to manipulate the data set.  We used mpg, weight and cylinders.  We also used model year 1976.
We used matplot lib to visualize the data.

# Conclusion:
As you can see, there is correlation between mpg, weight, and cylinders.  The higher mpg and less weight typically means less cylinders.  The kmeans algorithm does not perfecly cluster the data but it does a very good job.  We would like to develop a way to automatically determine the number of clusters in the future.
