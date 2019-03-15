import pandas as pd
from surprise import Reader, Dataset
from surprise import SVD, evaluate
import time

time_start =time.time()
ratings = pd.read_csv('movieratepredictions/train_ratings.csv') # reading data in pandas df
#ratings = pd.read_csv('dataset/ml-latest-small/ratings_testing.csv') # reading data in pandas df
# to load dataset from pandas df, we need `load_fromm_df` method in surprise lib

ratings_dict = {'itemID': list(ratings.movieId),
                'userID': list(ratings.userId),
                'rating': list(ratings.rating)}
df = pd.DataFrame(ratings_dict)

# A reader is still needed but only the rating_scale param is required.
# The Reader class is used to parse a file containing ratings.
reader = Reader(rating_scale=(0.5, 5.0))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

#Train

trainset = data.build_full_trainset()

# svd
algo = SVD(n_factors=30)
algo.fit(trainset)

testset = trainset.build_anti_testset()
print("Built anti test")
predictions = algo.test(testset)
print('Computation time: %.2f'%(time.time()-time_start))

with open('svd_100_collab_filtering.csv', 'w') as f:
    f.write("userId,movieId,r_ui,estimate,was_impossible\n")
    for item in predictions:
        f.write("{0},{1},{2},{3},{4}\n".format(item[0], item[1],item[2],item[3],item[4]))