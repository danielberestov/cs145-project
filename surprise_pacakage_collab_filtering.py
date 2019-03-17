import math
import os

import pandas as pd
from surprise import Reader, Dataset
from surprise import SVD, evaluate
import time

def get_predicted_dict(predictions):
    predicted_dict = {}
    for item in predictions:
        userId = int(float(item[0]))
        movieId = int(float(item[1]))
        est_rating = float(item[3])

        if userId not in predicted_dict:
            local_dict = {}
        else:
            local_dict = predicted_dict[userId]
        local_dict[movieId] = est_rating
        predicted_dict[userId] = local_dict

    return predicted_dict


def calculate_rating_for_batch(predictions, avg_usr_df, test_ratings_df, index_of_test_data_iterations):

    size_df = test_ratings_df.shape[0]

    predicted_dict = get_predicted_dict(predictions)
    max_user_id_in_predictions = max(predicted_dict, key=int)


    while index_of_test_data_iterations < size_df:
        id = int(test_ratings_df.loc[index_of_test_data_iterations]['ID'])
        user_id = int(test_ratings_df.loc[index_of_test_data_iterations]['userID'])
        movie_id = int(test_ratings_df.loc[index_of_test_data_iterations]['movieID'])

        if user_id > max_user_id_in_predictions:
            break

        predicted_movies = predicted_dict[user_id]
        if movie_id in predicted_movies:
            pred_rating = predicted_movies[movie_id]
            with open('sample_submission.csv', 'a') as f:
                f.write("{0},{1:.3f}\n".format(id, pred_rating))
        else:
            avg_user_rating = avg_usr_df.loc[user_id-1]['avg_rate']
            with open('sample_submission.csv', 'a') as f:
                f.write("{0},{1:.3f}\n".format(id, avg_user_rating))

        index_of_test_data_iterations += 1

    return index_of_test_data_iterations


time_start = time.time()
ratings = pd.read_csv('movieratepredictions/train_ratings.csv', engine='python') # reading data in pandas df
avg_user_ratings = pd.read_csv('dataset/user_rate.csv', engine='python')
test_ratings = pd.read_csv('movieratepredictions/test_ratings.csv', engine='python')

ratings_dict = {'itemID': list(ratings.movieId),
                'userID': list(ratings.userId),
                'rating': list(ratings.rating)}

avg_usr_rating_dict = {'userID': list(avg_user_ratings.userId),
                'avg_rate': list(avg_user_ratings.avg_rate)}

test_ratings_dict = {'ID': list(test_ratings.Id),
                    'userID': list(test_ratings.userId),
                     'movieID': list(test_ratings.movieId)}

avg_usr_df = pd.DataFrame(avg_usr_rating_dict)
test_ratings_df = pd.DataFrame(test_ratings_dict)

# A reader is still needed but only the rating_scale param is required.
# The Reader class is used to parse a file containing ratings.
reader = Reader(rating_scale=(0.5, 5.0))


df = pd.DataFrame(ratings_dict)
global_index = 0
num_users_per_batch = 1000
size_df = df.shape[0]
start_next_batch = True
starting_userId = 0
start_index = 0
index_of_test_data_iterations = 0

with open('sample_submission.csv', 'w') as f:
    f.write("Id,rating\n")

while global_index < size_df:
    user_id = int(df.loc[global_index]['userID'])
    if start_next_batch:
        starting_userId = user_id
        start_index = global_index
        start_next_batch = False

    if user_id - starting_userId == num_users_per_batch - 1 or global_index + 1 == size_df:

        while(user_id == int(df.loc[global_index]['userID'])):
            global_index += 1
        global_index -= 1

        batch_df = df[start_index:global_index]
        data = Dataset.load_from_df(batch_df[['userID', 'itemID', 'rating']], reader)

        trainset = data.build_full_trainset()

        # svd
        algo = SVD()
        algo.fit(trainset)

        testset = trainset.build_anti_testset()
        print("Built anti test")
        predictions = algo.test(testset)
        print('Computation time: %.2f' % (time.time() - time_start))
        print('Num rows from user id ' + str(starting_userId) + " to " + str(user_id) + " is " + str(global_index - start_index  + 1))
        print("Num items in batch: " + str(user_id - starting_userId + 1))

        index_of_test_data_iterations = calculate_rating_for_batch(predictions, avg_usr_df, test_ratings_df, index_of_test_data_iterations)

        start_next_batch = True
    global_index +=1