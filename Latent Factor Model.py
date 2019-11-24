from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import time
from numpy import linalg as LA

no_of_users = 610
no_of_movies = 10000
no_of_latent_factors = 15
gamma = 0.0002
lembda = 0.0003




utility_matrix = np.zeros((no_of_users,no_of_movies))
user_lfm =  np.random.random((no_of_users,no_of_latent_factors))
movie_lfm =  np.random.random((no_of_movies,no_of_latent_factors))


df = pd.read_csv('ratings10k.csv',  delim_whitespace=False, sep=',', header=None)
df = df.drop(df.index[0])
df = df.values.tolist()
# shuffle
random.Random(4).shuffle(df)
# random.Random(4).shuffle(df2)

# divide in test and train
train, test = train_test_split(df, test_size=0.2)







def createUtilityMatrix():
    for each in train:               # creating rating matrix
        userId = int(int(each[0]) - 1)
        movieId = int(int(each[1]) - 1)
        rating = float(each[2])
        utility_matrix[userId][movieId] = rating

def getRating(user,movie):
    return  float(np.dot(user_lfm[user],movie_lfm[movie]))

def update(user,movie,err):
    old_user = user_lfm[user]
    old_movie = movie_lfm[movie]
    user_lfm[user] += gamma*(err*(old_user) - lembda*(old_movie))
    movie_lfm[movie] += gamma * (err * (old_movie) - lembda * (old_user))

def iterate(no_of_iterations,printing):

    for i in range(no_of_iterations):
        print("iteration no", i)
        for user in range(len(utility_matrix)):
            for movie in range(len(utility_matrix[0])):
                if utility_matrix[user][movie] == 0:
                    continue
                rating = getRating(user, movie)
                # print("rating",rating)
                e = utility_matrix[user][movie] - rating
                if printing==1:
                    print("errpr", e)
                update(user, movie, e)

def testModel():
    RMSE = float(0)
    MAE = float(0)
    for each in test:               # creating rating matrix
        userId = int(int(each[0]) - 1)
        movieId = int(int(each[1]) - 1)
        rating = float(each[2])
        rating_cal = getRating(userId, movieId)
        e = abs(rating - rating_cal)
        RMSE += e**2
        MAE += e
    RMSE /= len(test)
    MAE /= len(test)
    return RMSE,MAE




createUtilityMatrix()

iterate(50,0)

a  = time.time()
rmse,mae = (testModel())
b = time.time()
print("error",rmse,mae)
print("Time Taken",b-a)


