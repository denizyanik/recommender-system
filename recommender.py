import sqlite3

import pandas
import numpy as np
import pdb
from tqdm import tqdm
np.random.seed(10)

train_small = pandas.read_csv(r"comp3208-train-small.csv",names = ["user","item","score","timestamp"])
#train = pandas.read_csv(r"comp3208-train.csv",names = ["user","item","score","timestamp"])

train_small = train_small[train_small['score'] != "rating"]
import warnings
warnings.filterwarnings("error")
train_small.score = train_small.score.apply(float)

n_factor = 40
gamma = 0.0002
lam = 0.02
n_iterations = 1000


conn = sqlite3.connect('comp3208_results.db')
c = conn.cursor()

n_users = train_small['user'].max()
n_items = train_small['item'].max()

#n_users1 = len(train['user'].unique())
#n_items1 = len(train['item'].unique())

#pdb.set_trace()

p = np.random.random((n_users,n_factor))
q = np.random.random((n_items,n_factor))


for i in range(0,n_iterations):
    sample = train_small.sample(frac=1).iterrows()

    for _,row in tqdm(train_small.sample(frac=1).iterrows()):
        try:
            u,i,r,_ = row
            user = int(u)-1
            item = int(i)-1

            predicted_rating = np.dot(p[user],q[item])


            error = int(r) - predicted_rating
            #print(gamma * (error * q[item, :] - lam * p[user, :]))
            p[user, :] += gamma * (error * q[item, :] - lam * p[user, :])
            #print(gamma * (error * p[user, :] - lam * q[item, :]))
            q[item, :] += gamma * (error * p[user, :] - lam * q[item, :])

        except RuntimeWarning:
            print(r)
            print(predicted_rating)
            pdb.set_trace()




    print(p)
    print(q)
