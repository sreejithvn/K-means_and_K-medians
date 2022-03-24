# K-mean
import pandas as pd
import numpy as np
from copy import deepcopy

SEED = np.random.randint(100)
np.random.seed(53)
# print(SEED)
# 53, 40, 18

# import data
data1 = pd.read_csv('CA2Data/animals', header=None, delimiter=' ')
data2 = pd.read_csv('CA2Data/countries', header=None, delimiter=' ')
data3 = pd.read_csv('CA2Data/fruits', header=None, delimiter=' ')
data4 = pd.read_csv('CA2Data/veggies', header=None, delimiter=' ')
# dataset = pd.concat([data1, data2, data3, data4])

# concatenate data points
dataset = pd.concat([data1.iloc[:,1:], data2.iloc[:,1:], data3.iloc[:,1:], data4.iloc[:,1:]])

# convert data set to numpy array
dataset = np.array(dataset)
dataset.shape
len(dataset)

def distance(X,Y):
    #Return the Euclidean distance between X and Y
    return np.linalg.norm(X-Y)

# choose 'k' for number of clusters 
k=4
num_centroids = k

# Initialise 'k' centroids (y1, .. yk) randomly from the data set
centroids = dataset[np.random.randint(dataset.shape[0], size=num_centroids), :]

# Define K clusters (C1, .. , Ck)


def check_convergence():
    pass


def get_centroid_positions():
    data_centroid_indices = np.zeros((len(dataset),1))

    num_datapoints = len(dataset)

    # loop through each datapoint
    for index in range(num_datapoints):
        distances = np.zeros((num_centroids, 1))
        for centroid_index in range(num_centroids):
            distance_to_centroid = distance(dataset[index, :], centroids[centroid_index, :])
            distances[centroid_index] = distance_to_centroid
        closest_centroid_position = np.argmin(distances)
        # Assign the datapoint to the cluster Ck corresponding to the one with the lowest distance
        data_centroid_indices[index] = closest_centroid_position
#     print(data_centroid_indices)
    return data_centroid_indices



# update centroids
for _ in range(10000):
    # if check_convergence():
    #     break
    num_centroids = k
    data_centroid_indices = get_centroid_positions()
    data = deepcopy(dataset)
    # print(data_centroid_indices.flatten())
    ## data[np.int8(data_centroid_indices) == [0]]
    for centroid_index in range(num_centroids):
    # find the eucledian distance from the datapoint to each centroid yk
    # Compute the mean of each cluster, and set them as the new centroids
        centroids[centroid_index] = np.mean(data[data_centroid_indices.flatten() == centroid_index], axis=0)
        # print(centroids[:,:])
        # print(data[data_centroid_indices.flatten() == centroid_index])
print("\nKMeans Centroids from computed for k = 4:")
print(centroids)


# ### TESTING THE CODE
from sklearn.cluster import KMeans

model = KMeans(n_clusters=k)
model.fit(data)
print("\n\nOutput from SKLEARN")
print(model.cluster_centers_)



# def cluster_plots(dataset, medoidInd=[], colours = 'gray', title = 'Dataset'):
#     fig,ax = plt.subplots()
#     fig.set_size_inches(12, 12)
#     ax.set_title(title,fontsize=14)
#     ax.set_xlim(min(dataset[:,0]), max(dataset[:,0]))
#     ax.set_ylim(min(dataset[:,1]), max(dataset[:,1]))
#     ax.scatter(dataset[:, 0], dataset[:, 1],s=8,lw=1,c= colours)

#     #Plot medoids if they are given
#     if len(medoidInd) > 0:
#         ax.scatter(dataset[medoidInd, 0], dataset[medoidInd, 1],s=8,lw=6,c='red')
#     fig.tight_layout()
#     plt.show()