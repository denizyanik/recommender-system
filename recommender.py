import sqlite3

import pandas
import numpy as np

train_small = pandas.read_csv(r"comp3208-train-small.csv",names = ["user","item","score","timestamp"])
n_factor = 40
gamma = 0.1
lam = 0
n_iterations = 1000


conn = sqlite3.connect('comp3208_results.db')
c = conn.cursor()

n_users = len(train_small['user'].unique())
n_items = len(train_small['item'].unique())

p = np.random.random((n_users,n_factor))
q = np.random.random((n_items,n_factor))


for i in range(0,n_iterations):

    for u,i,r,_ in train_small.sample(frac=1):
        predicted_rating = np.dot(p[u].T,q[i])

        error = r - predicted_rating


