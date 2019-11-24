from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import time
from numpy import linalg as LA

no_of_users = 610
no_of_movies = 10000
no_of_nearest_neighbours = 5




rating_matrix = np.zeros((no_of_users,no_of_movies))
pearson_matrix_ii = np.zeros((no_of_users,no_of_movies))
pearson_matrix_ii_test = np.zeros((no_of_users,no_of_movies))

user_user_similarity = {}


df = pd.read_csv('ratings10k.csv',  delim_whitespace=False, sep=',', header=None)
df = df.drop(df.index[0])
df = df.values.tolist()
# shuffle
random.Random(4).shuffle(df)
# random.Random(4).shuffle(df2)

# divide in test and train
train, test = train_test_split(df, test_size=0.2)


for each in test:               # creating rating matrix
    userId = int(int(each[0]) - 1)
    movieId = int(int(each[1]) - 1)
    rating = float(each[2])
    rating_matrix[userId][movieId] = rating

# print(rating_matrix)

def cosineSimilarity(vec1,vec2):
    score = float(np.dot(vec1,vec2))
    if score==0:
        return 0
    score /= LA.norm(vec1)
    score /= LA.norm(vec2)
    return score




def calSimUsers():
    for user1 in range(len(pearson_matrix_ii)):
        print("similarity movie no",user1)
        ls = []
        for user2 in range(len(pearson_matrix_ii)):
            ls.append(float(cosineSimilarity(pearson_matrix_ii[user1],pearson_matrix_ii[user2])))
            # item_item_similarity[(movie1,movie2)] = float(cosineSimilarity(pearson_matrix_ii[movie1],pearson_matrix_ii[movie2]))
            # item_item_similarity[(movie2,movie1)] = item_item_similarity[(movie1,movie2)]
        user_user_similarity[user1] = ls


def getIndex(user):
    user = int(user)
    ls = user_user_similarity.get(user)
    if ls == None:
        return -1
    if np.sum(ls)==0:
        return -1
    ans = []
    ls_cpy = ls[:]
    ls_cpy.sort(reverse=True)
    for i in range(no_of_nearest_neighbours):
        ans.append(ls.index(ls_cpy[i]))
    return ans




pearson_matrix_ii[:] = rating_matrix
pearson_matrix_ii_test[:] = rating_matrix
pearson_matrix_ii_test = np.array(pearson_matrix_ii_test)

pearson_matrix_ii = np.array(pearson_matrix_ii)
print(pearson_matrix_ii.shape)
# pearson_matrix_ii = pearson_matrix_ii.transpose()
print(pearson_matrix_ii.shape)

print(pearson_matrix_ii)
user_mean = np.mean(pearson_matrix_ii, axis = 1, dtype=np.float)
# movie_mean = np.sum(pearson_matrix_ii, axis = 1, dtype=np.float)
# for movie in range(len(pearson_matrix_ii)):
#     count = 0
#     for user in range(len(pearson_matrix_ii[0])):
#         if pearson_matrix_ii[movie][user]==0:
#             continue
#         else:
#             count += 1
#     if count!=0:
#         print("count",count)
#         movie_mean[movie] /= count

print(user_mean)
# print(movie_mean1)

a = time.time()
for i in range(no_of_users):
    print("centering movie",i)
    for j in range(no_of_movies):
        if pearson_matrix_ii[i][j] != 0:
            pearson_matrix_ii[i][j] -= user_mean[i]
b = time.time()
print(pearson_matrix_ii)

print("time taken",b-a)


# print(item_item_similarity)
calSimUsers()
# print(item_item_similarity)
RMSE = float(0)
MAE = float(0)
count = 0

a = time.time()

# ls1 = item_item_similarity.keys()
# print(ls1)
# ls2 = []

for i in range(len(test)):
    user = int(int(test[i][0])-1)
    # ls2.append(movie)
    indices = getIndex(user)
    if indices==-1:
        continue
    print("found")
    sim = user_user_similarity[user]
    print("Final movie",user)
    for j in range(no_of_movies):
        if pearson_matrix_ii_test[user][j]==0:
            continue
        numerator = float(0)
        denominator = float(0)
        for index in range(len(indices)):
            numerator += pearson_matrix_ii_test[index][j] * sim[index]
            denominator += sim[index]
        # print(numerator)
        # print(denominator)
        if numerator==0 or denominator==0:
            # calculated_rating = 0
            continue
        else:
            calculated_rating = numerator / denominator
        e = pearson_matrix_ii_test[user][j] - calculated_rating
        RMSE += e**2
        MAE += abs(e)
        count +=1
b = time.time()
print(b-a)

RMSE /= count
RMSE = np.sqrt(RMSE)
MAE = MAE/count
print("RMSE",RMSE)
print("MAE",MAE)





# print(ls2)








