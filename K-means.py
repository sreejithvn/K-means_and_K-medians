# K-mean * K-medians


def euclidean_distance(X,Y):
    # Return the Euclidean distance between X and Y
    return np.linalg.norm(X-Y)


def manhattan_distance(X,Y):
    # Return the Manhattan distance between X and Y
    # return np.abs(X-Y).sum()
    return np.sum(np.abs(X-Y))


def check_convergence(k_means, k_medians):

    # if k_means:
    #     np.sum()
    pass


def get_centroid_positions(k_means, k_medians):
    clusters = np.zeros((len(dataset),1))

    num_datapoints = len(dataset)

    # loop through each datapoint
    for index in range(num_datapoints):
        distances = np.zeros((num_centroids, 1))
        for centroid_index in range(num_centroids):
            if k_means:
                # find the eucledian distance from the datapoint to each centroid yk
                distance_to_centroid = euclidean_distance(dataset[index, :], centroids[centroid_index, :])
                distances[centroid_index] = distance_to_centroid

            elif k_medians:
                # find the manhattan distance from the datapoint to each centroid yk
                distance_to_centroid = manhattan_distance(dataset[index, :], centroids[centroid_index, :])
                distances[centroid_index] = distance_to_centroid
        closest_centroid_index = np.argmin(distances)
        # Assign the datapoint to the cluster Ck corresponding to the one with the lowest distance
        clusters[index] = closest_centroid_index
#     print(clusters)
    return clusters


if __name__ == '__main__':
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


    # choose 'k' for number of clusters 
    k=4
    num_centroids = k

    # Initialise 'k' centroids (y1, .. yk) randomly from the data set
    centroids = dataset[np.random.randint(dataset.shape[0], size=num_centroids), :]

    # To only run K_means
    k_means = True
    k_medians = False

    # To only run K_medians
    # k_means = False
    # k_medians = True


    ### update centroids

    for _ in range(1000):
        # if check_convergence():
        #     break
        num_centroids = k
        clusters = get_centroid_positions(False, True)
        data = deepcopy(dataset)
        # print(clusters.flatten())
        for centroid_index in range(num_centroids):
            if k_means:
                # Compute the mean of each cluster, and set them as the new centroids
                centroids[centroid_index] = np.mean(data[clusters.flatten() == centroid_index], axis=0)
            elif k_medians:
                # Compute the median of each cluster, and set them as the new centroids
                centroids[centroid_index] = np.median(data[clusters.flatten() == centroid_index], axis=0)
            # print(centroids[:,:])
            # print(data[clusters.flatten() == centroid_index])
    print("\nKMeans Centroids from computed for k = 4:")
    print(centroids)

    #### TESTING THE CODE

    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k)
    model.fit(data)
    print("\n\nOutput from SKLEARN")
    print(model.cluster_centers_)


    # from pyclustering.cluster.kmedians import kmedians
    
    # # Create instance of K-Medians algorithm.
    # kmedians_instance = kmedians(data, centroids)
    # kmedians_instance.process()
    # median_clusters = kmedians_instance.get_clusters()
    # medians = kmedians_instance.get_medians()
    # print('median clusters', median_clusters)
    # print('pyclustering medians', medians)



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


