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


def calculate_validation_mse(val_ratings, predictions):
    num_samples = 0.0
    mse = 0.0
    ratings_dict = {'itemID': list(val_ratings.movieId),
                    'userID': list(val_ratings.userId),
                    'rating': list(val_ratings.rating)}
    df = pd.DataFrame(ratings_dict)

    size_df = df.shape[0]
    global_index = 0

    predicted_dict = get_predicted_dict(predictions)


    while global_index < size_df:
        user_id = int(df.loc[global_index]['userID'])
        movie_id = int(df.loc[global_index]['itemID'])
        rating = float(df.loc[global_index]['rating'])

        if user_id in predicted_dict:
            predicted_movies = predicted_dict[user_id]
        else:
            global_index += 1
            continue
        if movie_id in predicted_movies:
            pred_rating = predicted_movies[movie_id]
            num_samples += 1.0
            mse += math.pow(rating - pred_rating, 2)
        global_index +=1

    if num_samples > 0.0:
        mse /= num_samples
    print("Mean squared error for validation is: " + str(mse))


time_start = time.time()
ratings = pd.read_csv('dataset/ml-latest-small/full_ratings_test.csv', engine='python') # reading data in pandas df
#val_ratings = pd.read_csv('dataset/ml-latest-small/full_val_ratings_test.csv', engine='python')

ratings_dict = {'itemID': list(ratings.movieId),
                'userID': list(ratings.userId),
                'rating': list(ratings.rating)}

reader = Reader(rating_scale=(0.5, 5.0))
# A reader is still needed but only the rating_scale param is required.
# The Reader class is used to parse a file containing ratings.

df = pd.DataFrame(ratings_dict)
global_index = 0
num_users_per_batch = 1000
size_df = df.shape[0]
start_next_batch = True
starting_userId = 0
start_index = 0

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

        #calculate_validation_mse(val_ratings, predictions)

        if os.path.exists('svd_100_collab_filtering.csv'):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        with open('svd_100_collab_filtering.csv', append_write) as f:
            f.write("userId,movieId,estimate")
            for item in predictions:
                f.write("{0},{1},{2}\n".format(item[0], item[1], item[3]))

        start_next_batch = True
    global_index +=1