# K-mean * K-medians

def get_user_choice():
    """Gets user choice for choosing binary and multi-class perceptron, 
    and reporting accuracies for different classes of test and train data
    """
    while True:
        try:
            print('\n\nPlease enter your choice from either 1 - 4: \n')
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
    c = np.linalg.norm(data,axis=1).reshape(-1,1)
    return (1/c)*data

    
def euclidean_distance(X,Y):
    # Return the Euclidean distance between X and Y
    # return np.sqrt(np.sum((X-Y)**2))
    return np.linalg.norm(X-Y)


def manhattan_distance(X,Y):
    # Return the Manhattan distance between X and Y
    # return np.abs(X-Y).sum()
    return np.sum(np.abs(X-Y))


def check_convergence(k_means, k_medians):

    # if k_means:
    #     np.sum()
    pass


def fit_model(k, MAX_ITER):
    np.random.seed(53)
    
    num_centroids = k
    # print(f'k: {k}')
    # Initialise 'k' centroids (y1, .. yk) randomly from the data set
    centroids = data[np.random.randint(data.shape[0], size=num_centroids), :]
    # print(f'length centroids {len(centroids)}')
    # print('Initial centroids\n', centroids[:,:5])

    for _ in range(MAX_ITER):
        # if check_convergence():
        #     break

        # for the current iteration, get the centroids (index value) assigned to the corresponding datapoints index
        clusters = group_data_to_cluster(centroids)
        # get updated centroids for current iteration
        centroids = update_centroids(clusters, centroids)

    return clusters, centroids


def group_data_to_cluster(centroids):
    num_centroids = len(centroids)
    num_datapoints = len(data)
    # Initialise 'clusters', to later store the centroid index value to the corresponding datapoint index
    clusters = np.zeros((num_datapoints, 1))

    # loop through each datapoint
    for index in range(num_datapoints):
        # Initialise 'distances', to store the 'distance' value from each centroid to the datapoint
        distances = np.zeros((num_centroids, 1))
        # Loop through each centroid
        for centroid_index in range(num_centroids):
            if k_means:
                # for k-means, find the eucledian distance from the datapoint to each centroid
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
#     print(clusters)
    return clusters


def update_centroids(clusters, centroids):
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
    num_datapoints = len(data)
    precision = np.zeros((num_datapoints))
    recall = np.zeros((num_datapoints))
    f_score = np.zeros((num_datapoints))
    for index in range(num_datapoints):
        # get the count of the 'category' from it's assigned 'cluster'
        category_in_cluster_count = np.count_nonzero(category[clusters==clusters[index]] == category[index])
        # get the total count of datapoints belonging to the 'category' in the dataset
        category_total_count = np.count_nonzero(category==category[index])
        # get the count of datapoints assigned to the 'cluster'
        cluster_elements_count = np.count_nonzero(clusters == clusters[index])
    #     count = np.sum(cat[clust==clust[index]]==category[index])
        # compute precision
        precision[index] = category_in_cluster_count / cluster_elements_count
        recall[index] = category_in_cluster_count / category_total_count
    #     f_score[index] = 2*precision[index]*recall[index]/(precision[index]+recall[index])
    #     print(f'Count of category {category[index]} in cluster {clust[index]} = {category_in_cluster_count}')
    #     print(f'Count of category {category[index]} in dataset = {category_total_count}')
    #     print(f'Total elements in cluster {clust[index]} = {cluster_elements_count}')
    f_score = 2*precision*recall / (precision+recall)
    # print(precision)
    # print(recall)
    # print(f_score)
    precision = np.round((np.sum(precision) / num_datapoints), 2)
    recall = np.round((np.sum(recall) / num_datapoints), 2)
    f_score = np.round((np.sum(f_score) / num_datapoints), 2)
    # print(f'precision: {precision}')
    # print(f'recall:    {recall}')
    # print(f'f_score:   {f_score}')
    return precision, recall, f_score


def plot_metrics(precisions, recalls, f_scores):
    k_choices = np.arange(1,K_MAX+1)
    plt.plot(k_choices, precisions, label='precision')
    plt.plot(k_choices, recalls, label='recall')
    plt.plot(k_choices, f_scores, label='f_score')
    plt.xlabel('K')
    plt.ylabel('Metrics')
    plt.xticks(np.arange(1,K_MAX+1))
    plt.legend()
    if l2_len_norm:
        plt.title("B-CUBED Metrics for {alg}\n with l2 length normalised data".format(alg="k-means" if k_means == True else "k-medians"))
    else:
        plt.title("B-CUBED Metrics for {alg}\n without l2 length normalised data".format(alg="k-means" if k_means == True else "k-medians"))
    plt.show()

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from copy import deepcopy

    # SEED = np.random.randint(100)
    # np.random.seed(53)
    # print(SEED)
    # 53, 40, 18

    MAX_ITER = 10
    K_MAX = 9

    # ****** change PATH *******************

    # import data
    animals = pd.read_csv('CA2Data/animals', header=None, delimiter=' ')
    countries = pd.read_csv('CA2Data/countries', header=None, delimiter=' ')
    fruits = pd.read_csv('CA2Data/fruits', header=None, delimiter=' ')
    veggies = pd.read_csv('CA2Data/veggies', header=None, delimiter=' ')

    # Assign numerical 'category' values corresponding to true labels
    animals[0] = 0
    countries[0] = 1
    fruits[0] = 2
    veggies[0] = 3

    # concatenate data points
    data = pd.concat([animals, countries, fruits, veggies], ignore_index=True)

    # store category data to compute metrics
    # category = np.array(data[0])
    category = np.array(data[0]).reshape(-1,1)

    # drop category column, before fitting data to model
    data.drop(columns=[0], inplace=True)

    # convert data set to numpy array
    dataset = np.array(data)


    precisions = np.zeros(K_MAX)
    recalls = np.zeros(K_MAX)
    f_scores = np.zeros(K_MAX)

    while True:
        # reset data after each user choice
        data = deepcopy(dataset)

        user_choice = get_user_choice()

        if user_choice == '1':
            # Run K_means without l2 length normalised data
            k_means = True
            k_medians = False
            l2_len_norm = False
            if l2_len_norm:
                data = normalise_data(data)
            for k in range(1, K_MAX+1):
                clusters, _ = fit_model(k, MAX_ITER)
                precisions[k-1], recalls[k-1], f_scores[k-1] = compute_metrics(clusters, category)
            print(pd.DataFrame((precisions, recalls, f_scores), index=['precision', 'recall', 'f-score'], columns=[np.arange(1,K_MAX+1)]))
            plot_metrics(precisions, recalls, f_scores)


        elif user_choice == '2':
            # Run K_means with l2 length normalised data
            k_means = True
            k_medians = False
            l2_len_norm = True
            if l2_len_norm:
                data = normalise_data(data)               
            for k in range(1, K_MAX+1):
                clusters, _ = fit_model(k, MAX_ITER)
                precisions[k-1], recalls[k-1], f_scores[k-1] = compute_metrics(clusters, category)
            print(pd.DataFrame((precisions, recalls, f_scores), index=['precision', 'recall', 'f-score'], columns=[np.arange(1,K_MAX+1)]))
            plot_metrics(precisions, recalls, f_scores)

        elif user_choice == '3':
            # Run K_medians without l2 length normalised data
            k_means = False
            k_medians = True
            l2_len_norm = False
            if l2_len_norm:
                data = normalise_data(data)
            for k in range(1, K_MAX+1):
                clusters, _ = fit_model(k, MAX_ITER)
                precisions[k-1], recalls[k-1], f_scores[k-1] = compute_metrics(clusters, category)
            print(pd.DataFrame((precisions, recalls, f_scores), index=['precision', 'recall', 'f-score'], columns=[np.arange(1,K_MAX+1)]))
            plot_metrics(precisions, recalls, f_scores)

        elif user_choice == '4':
            # Run K_medians with l2 length normalised data
            k_means = False
            k_medians = True
            l2_len_norm = True
            if l2_len_norm:
                data = normalise_data(data)
            for k in range(1, K_MAX+1):
                clusters, _ = fit_model(k, MAX_ITER)
                precisions[k-1], recalls[k-1], f_scores[k-1] = compute_metrics(clusters, category)
            print(pd.DataFrame((precisions, recalls, f_scores), index=['precision', 'recall', 'f-score'], columns=[np.arange(1,K_MAX+1)]))
            plot_metrics(precisions, recalls, f_scores)


    #### update centroids based on new cluster data

    # print(centroids)

    #### ******************* IGNORE CODE BELOW ********************

    #### TESTING THE CODE

    # from sklearn.cluster import KMeans

    # model = KMeans(n_clusters=k)
    # model.fit(data)
    # print("\n\nOutput from SKLEARN")
    # print(model.cluster_centers_)


    # from pyclustering.cluster.kmedians import kmedians
    
    # # Create instance of K-Medians algorithm.
    # kmedians_instance = kmedians(data, centroids)
    # kmedians_instance.process()
    # median_clusters = kmedians_instance.get_clusters()
    # medians = kmedians_instance.get_medians()
    # print('median clusters', median_clusters)
    # print('pyclustering medians', medians)


