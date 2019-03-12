from __future__ import print_function

import numpy as np
import os
from numpy import genfromtxt

def load_movie_tag_ids(ROOT, file_name):
  file_name = os.path.join(ROOT, file_name)
  my_data = genfromtxt(file_name, delimiter=',', skip_header=1)
  return my_data

def create_movie_tag_ids_vector_csv(ROOT):

    # This function returns a mapping of a movie id, to the index it will belong to once the csv
    # becomes a numpy array

    dic = {}

    output_file = open('dataset/genome-scores-processed.csv', 'w')
    file_name = os.path.join(ROOT, 'genome-scores.csv')
    with open(file_name) as f:
        #read first line, get rid of string column identifiers on the first line
        f.readline()
        output_file.write("movieId")
        for x in range(1,1129):
            output_file.write(",tagId" + str(x))
        movieId = -1
        dict_index = -1
        for x in f:
            x = x.split(",")
            if movieId != x[0]:
                movieId = x[0]
                output_file.write("\n" + str(x[0]) + "," + x[2].replace("\n", ""))
                dict_index += 1
                dic[int(movieId)] = dict_index
            else:
                output_file.write("," + x[2].replace("\n", ""))
    output_file.close()

    return dic

def create_movie_tag_ids_vector_from_test_csv(ROOT, file_to_write_to, csv_to_read_from, movie_tag_ids_vector, dict_mapping):

    output_file = open(file_to_write_to, 'w')
    file_name = os.path.join(ROOT, csv_to_read_from)
    with open(file_name) as f:
        # read first line, get rid of string column identifiers on the first line
        f.readline()
        output_file.write("movieId")
        for x in range(1,1129):
            output_file.write(",tagId" + str(x))
            if x == 1128:
                output_file.write("\n")

        duplicate_dict = {}
        for x in f:
            x = x.split(",")
            dict_key = int(float(x[1]))

            if dict_key not in duplicate_dict:
                if dict_key in dict_mapping:
                    duplicate_dict[dict_key] = 1
                    numpy_index = dict_mapping.get(int(float(x[1])))
                    np.savetxt(output_file, movie_tag_ids_vector[numpy_index][np.newaxis], delimiter=',')
    output_file.close()

def compute_L2_distances_vectorized(X_train, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train WITHOUT using any for loops.

    Inputs:
    - X: A numpy array of shape (num_test, 1128) containing all the relevance scores for the 1128 tags for
    any movie Id

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    M = np.dot(X, (X_train).T)
    test_square = np.square(X).sum(axis=1)[:, np.newaxis]
    train_square = np.square(X_train).sum(axis=1)
    dists = np.sqrt(test_square - (2 * M) + train_square)
    pass

    return dists

def get_closest_related_movie_ids(dists, np_Train, k=1):
    """
    Given a matrix of distances between test points and training points,
    return the k number of movie ids that are close to the test movieIds

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - closestRelatedMovieIds: A numpy array of shape (num_test,k) containing the movie ids
    for the closest related movie to the movie at each index inside the X array from compute_L2_distances_vectorized
    """

    num_test = dists.shape[0]
    closestRelatedMovieIds = np.zeros([num_test, k])
    for i in np.arange(num_test):
        np_Train_index = np.argsort(dists[i], kind='mergesort')[:k].astype(int)
        zeroes = np.zeros(k).astype(int)
        closestRelatedMovieIds[i] = np_Train[np_Train_index, zeroes]
    return closestRelatedMovieIds