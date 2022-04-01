
"""K-mean and K-medians clustering algorithms"""


def get_user_choice():
    """Gets user choice for selecting clustering algorithm and enabling l2 length data normalisation
    """
    while True:
        try:
            print('\n\nPlease enter your choice from either 1 - 5: \n')
            choice = input('''    1. Q3 - Plot B-cubed Precision, Recall, F-score for K-means (k = 1-9)
    2. Q4 - Plot B-cubed Precision, Recall, F-score for K-means for l2-length normalised data (k = 1-9)
    3. Q5 - Plot B-cubed Precision, Recall, F-score for K-medians (k = 1-9)
    4. Q6 - Plot B-cubed Precision, Recall, F-score for K-medians for l2-length normalised data (k = 1-9)
    5. Quit\n\n''')
            if choice in '12345' and len(choice) == 1:
                if choice == '5':
                    exit()
                return choice
            else:
                raise ValueError
        except ValueError:
            print(f"Your choice '{choice}' is invalid\n")


def normalise_data(data):
    """Function to normalise each data point to unit l2 length

    Args:
        data (numpy array): dataset to be normalised

    Returns:
        numpy array: l2 length normalised data
    """
    # compute the length of each datapoint (vector)
    c = np.linalg.norm(data, axis=1).reshape(-1,1)
    # return l2 length normalised (to unit length) data
    return (1/c)*data

    
def euclidean_distance(X,Y):
    """Compute squared euclidean distance between the vectors, X and Y

    Args:
        X (numpy array): data point as 1D numpy array
        Y (numpy array): centroid as 1D numpy array

    Returns:
        float: squared euclidean distance
    """
    # Return the squared Euclidean distance between X and Y
    return np.sum((X-Y)**2)


def manhattan_distance(X,Y):
    """Compute manhattan distance between the vectors, X and Y

    Args:
        X (numpy array): data point as 1D numpy array
        Y (numpy array): centroid as 1D numpy array

    Returns:
        float: manhattan distance
    """
    # Return the Manhattan distance between X and Y
    return np.sum(np.abs(X-Y))


def fit_model(k, MAX_ITER):
    """Fit the model with the data, and compute the centroids and assign data points to 
    clusters for each choice of 'k' (cluster count)

    Args:
        k (integer): cluster count
        MAX_ITER (integer): maximum number of iterations to run for updating centroids and 
        clusters, before cluster convergence

    Returns:
        clusters (numpy array): array with the final centroids index value mapped to 
        index of corresponding datapoint, after cluster convergence
        centroids (numpy array): array with centroids, after cluster convergence
    """
    # set random seed
    np.random.seed(53)
    
    num_centroids = k
    # Initialise 'k' centroids (y1, .. yk) randomly from the data set
    centroids = data[np.random.randint(data.shape[0], size=num_centroids), :]
    # initialise 'clusters' to None
    clusters = None
    # iterate to update the centroids, and group the datapoints into clusters, until convergence.
    for _ in range(MAX_ITER):
        # store clusters in 'old_clusters', to later check for convergence of clusters after updation
        old_clusters = clusters
        # for the current iteration, get the centroids (index value) assigned to the corresponding datapoints index
        clusters = group_data_to_cluster(centroids)
        # check for convergence, and exit if there is no change in assigned datapoints to each cluster
        if np.all(old_clusters == clusters):
            break
        # get updated centroids for current iteration
        centroids = update_centroids(clusters, centroids)

    return clusters, centroids


def group_data_to_cluster(centroids):
    """Assigns each data points to their respective clusters, based on the shortest distance 
    to centroid (based on the choice of clustering algorithm)

    Args:
        centroids (numpy array): current updated centroids

    Returns:
        clusters (numpy array): array with the updated closest centroids, index value 
        mapped to index of corresponding datapoint
    """
    num_centroids = len(centroids)
    num_datapoints = len(data)
    # Initialise 'clusters', to later store the centroid index value to the corresponding datapoint index
    clusters = np.zeros((num_datapoints, 1))

    # loop through each datapoint
    for index in range(num_datapoints):
        # Initialise 'distances', to store the 'distance' value from each centroid to the datapoint
        distances = np.zeros((num_centroids, 1))
        # loop through each centroid
        for centroid_index in range(num_centroids):
            if k_means:
                # for k-means, find the 'squared' eucledian distance from the datapoint to each centroid
                distance_to_centroid = euclidean_distance(data[index], centroids[centroid_index])
                # update the 'distance' value to the 'distances' array
                distances[centroid_index] = distance_to_centroid

            elif k_medians:
                # for k-medians, find the manhattan distance from the datapoint to each centroid
                distance_to_centroid = manhattan_distance(data[index], centroids[centroid_index])
                # update the 'distance' value to the 'distances' array
                distances[centroid_index] = distance_to_centroid

        # Get the closest centroids index value
        closest_centroid_index = np.argmin(distances)
        # Assign the closest centroid index value to the corresponding datapoint index (in the 'clusters' array)
        clusters[index] = closest_centroid_index

    return clusters


def update_centroids(clusters, centroids):
    """update the centroids using the current datapoint distribution in clusters 
    (based on the choice of clustering algorithm)

    Args:
        clusters (numpy array): array with the current datapoint distribution,
        with each centroid index value mapped to the respective data point index
        centroids (numpy array): array with previous centroids

    Returns:
        centroids (numpy array): array with updated centroids
    """
    num_centroids = len(centroids)
    # Iterate over the centroids, to update them based on the updated cluster data
    for centroid_index in range(num_centroids):
        if k_means:
            # Compute the mean of datapoints for each cluster, and set them as the new centroids
            centroids[centroid_index] = np.mean(data[clusters.flatten() == centroid_index], axis=0)
        elif k_medians:
            # Compute the median of datapoints for each cluster, and set them as the new centroids
            centroids[centroid_index] = np.median(data[clusters.flatten() == centroid_index], axis=0)

    return centroids


def compute_metrics(clusters, category):
    """Compute B-cubed metrics for the dataset

    Args:
        clusters (numpy array): array with the current datapoint distribution,
        with each centroid index value mapped to the respective data point index
        category (numpy array): array with category (true label) values for the dataset

    Returns:
        (floats): computed B-cubed metric scores
    """
    num_datapoints = len(data)
    # initialise arrays for storing B-cubed metrics for each datapoint
    precision = np.zeros((num_datapoints))
    recall = np.zeros((num_datapoints))
    f_score = np.zeros((num_datapoints))
    # iterate over each datapoint, and compute it's individual B-cubed scores
    for index in range(num_datapoints):
        # get the count of the 'category' from it's assigned 'cluster', corresponding to this datapoint
        category_in_cluster_count = np.count_nonzero(category[clusters==clusters[index]] == category[index])
        # get the total count of datapoints belonging to the 'category' (true label), corresponding to this datapoint, in the dataset
        category_total_count = np.count_nonzero(category==category[index])
        # get the count of datapoints assigned to the 'cluster', to which this datapoint belongs
        cluster_elements_count = np.count_nonzero(clusters == clusters[index])
        # compute precision for this datapoint
        precision[index] = category_in_cluster_count / cluster_elements_count
        # compute recall for this datapoint
        recall[index] = category_in_cluster_count / category_total_count
    # compute f_score's for the each datapoints in the data set
    f_score = 2*precision*recall / (precision+recall)
    # compute overall scores for the entire dataset, for the corresponding 'k' chosen
    precision = np.round((np.sum(precision) / num_datapoints), 2)
    recall = np.round((np.sum(recall) / num_datapoints), 2)
    f_score = np.round((np.sum(f_score) / num_datapoints), 2)

    return precision, recall, f_score


def plot_metrics(precisions, recalls, f_scores):
    """Plot the B-cubed metric scores for each choice of clustering algorithm

    Args:
        precisions (numpy array): precisions for dataset corresponding to each value of 'k'
        recalls (numpy array): recalls for dataset corresponding to each value of 'k'
        f_scores (numpy array): f_scores for dataset corresponding to each value of 'k'
    """
    # create an array of 'k' values for x-axis
    k_choices = np.arange(1,K_MAX+1)
    plt.plot(k_choices, precisions, label='precision')
    plt.plot(k_choices, recalls, label='recall')
    plt.plot(k_choices, f_scores, label='f_score')
    plt.xlabel('K')
    plt.ylabel('B-CUBED metric scores')
    plt.xticks(np.arange(1,K_MAX+1))
    plt.legend()
    # Add annotations to the plot
    for i,j in zip(k_choices, precisions):
        plt.annotate(str(j), xy=(i,j))
    for i,j in zip(k_choices, recalls):
        plt.annotate(str(j), xy=(i,j))
    for i,j in zip(k_choices, f_scores):
        plt.annotate(str(j), xy=(i,j))
    if l2_len_norm:
        plt.title("B-CUBED Metrics for {alg}\n with l2 length normalised data".format(alg="k-means" if k_means == True else "k-medians"))
    else:
        plt.title("B-CUBED Metrics for {alg}\n without l2 length normalised data".format(alg="k-means" if k_means == True else "k-medians"))
    plt.show()


if __name__ == '__main__':
    # import the necessary libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from copy import deepcopy

    MAX_ITER = 100
    K_MAX = 9

    # import data and convert them to numpy arrays
    animals = pd.read_csv('animals', header=None, delimiter=' ').to_numpy()
    countries = pd.read_csv('countries', header=None, delimiter=' ').to_numpy()
    fruits = pd.read_csv('fruits', header=None, delimiter=' ').to_numpy()
    veggies = pd.read_csv('veggies', header=None, delimiter=' ').to_numpy()

    # Assign numerical 'category' values corresponding to true labels
    animals[:,0] = 0
    countries[:,0] = 1
    fruits[:,0] = 2
    veggies[:,0] = 3

    # store category data to compute B-cubed metrics
    category = np.hstack((animals[:,0], countries[:,0], fruits[:,0], veggies[:,0])).reshape(-1,1).astype(int)

    # convert data set to numpy array
    dataset = np.vstack((animals[:,1:], countries[:,1:], fruits[:,1:], veggies[:,1:])).astype(float)

    # initialise arrays to store B-cubed metrics for each value of 'k' (cluster count)
    precisions = np.zeros(K_MAX)
    recalls = np.zeros(K_MAX)
    f_scores = np.zeros(K_MAX)

    while True:
        # reset data after each user choice
        data = deepcopy(dataset)

        # get user choice to select the clustering algorithm, and l2 length normalisation choice
        user_choice = get_user_choice()

        if user_choice == '1':
            # Run K_means without l2 length normalised data
            k_means = True
            k_medians = False
            l2_len_norm = False

            # Check if l2 length normalisation required
            if l2_len_norm:
                data = normalise_data(data)

            # compute the clusters, centroids and B-cubed metric scores for each 'k' value, and finally plot them 
            for k in range(1, K_MAX+1):
                clusters, centroids = fit_model(k, MAX_ITER)
                precisions[k-1], recalls[k-1], f_scores[k-1] = compute_metrics(clusters, category)
            print(pd.DataFrame((precisions, recalls, f_scores), index=['precision', 'recall', 'f-score'], columns=[np.arange(1,K_MAX+1)]))
            plot_metrics(precisions, recalls, f_scores)


        elif user_choice == '2':
            # Run K_means with l2 length normalised data
            k_means = True
            k_medians = False
            l2_len_norm = True

            # Check if l2 length normalisation required
            if l2_len_norm:
                data = normalise_data(data) 
                
            # compute the clusters, centroids and B-cubed metric scores for each 'k' value, and finally plot them 
            for k in range(1, K_MAX+1):
                clusters, centroids = fit_model(k, MAX_ITER)
                precisions[k-1], recalls[k-1], f_scores[k-1] = compute_metrics(clusters, category)
            print(pd.DataFrame((precisions, recalls, f_scores), index=['precision', 'recall', 'f-score'], columns=[np.arange(1,K_MAX+1)]))
            plot_metrics(precisions, recalls, f_scores)

        elif user_choice == '3':
            # Run K_medians without l2 length normalised data
            k_means = False
            k_medians = True
            l2_len_norm = False

            # Check if l2 length normalisation required
            if l2_len_norm:
                data = normalise_data(data)

            # compute the clusters, centroids and B-cubed metric scores for each 'k' value, and finally plot them 
            for k in range(1, K_MAX+1):
                clusters, centroids = fit_model(k, MAX_ITER)
                precisions[k-1], recalls[k-1], f_scores[k-1] = compute_metrics(clusters, category)
            print(pd.DataFrame((precisions, recalls, f_scores), index=['precision', 'recall', 'f-score'], columns=[np.arange(1,K_MAX+1)]))
            plot_metrics(precisions, recalls, f_scores)

        elif user_choice == '4':
            # Run K_medians with l2 length normalised data
            k_means = False
            k_medians = True
            l2_len_norm = True

            # Check if l2 length normalisation required
            if l2_len_norm:
                data = normalise_data(data)

            # compute the clusters, centroids and B-cubed metric scores for each 'k' value, and finally plot them 
            for k in range(1, K_MAX+1):
                clusters, centroids = fit_model(k, MAX_ITER)
                precisions[k-1], recalls[k-1], f_scores[k-1] = compute_metrics(clusters, category)
            print(pd.DataFrame((precisions, recalls, f_scores), index=['precision', 'recall', 'f-score'], columns=[np.arange(1,K_MAX+1)]))
            plot_metrics(precisions, recalls, f_scores)