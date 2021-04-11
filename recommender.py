import sqlite3

import pandas
import numpy as np
from tqdm import tqdm
np.random.seed(10)


class RecommenderSystem:

    def __init__(self, k=40, gamma=0.001, lam=0, db=sqlite3.connect('comp3208_example.db')):
        self.k = k
        self.gamma = gamma
        self.lam = lam
        self.db = db.cursor()

        # Construct P and Q matrices
        self.db.execute('SELECT MAX(UserID) FROM example_table')
        n_users = self.db.fetchone()[0]
        self.db.execute('SELECT MAX(ItemID) FROM example_table')
        n_items = self.db.fetchone()[0]

        self.p = np.random.uniform(max=5/k, size=(n_users, k))
        self.q = np.random.uniform(max=5/k, size=(n_items, k))

    def get_shuffled_rows(self):
        self.db.execute('SELECT * FROM example_table ORDER BY RANDOM()')
        for row in self.db:
            yield row

    def stochastic_gradient_descent(self, iterations):
        for i in range(iterations):
            for row in self.get_shuffled_rows():
                u, i, r, _ = row
                u = u - 1
                i = i - 1

                predicted_rating = np.dot(self.p[u], self.q[i])

                error = r - predicted_rating
                self.p[u, :] += self.gamma * (error * self.q[i, :] - self.lam * self.p[u, :])
                self.q[i, :] += self.gamma * (error * self.p[u, :] - self.lam * self.q[i, :])

    def predict(self, user, item):
        return self.p[user, :].dot(self.q[item, :].T)


recommender = RecommenderSystem()


"""
train_small = pandas.read_csv(r"comp3208-train-small.csv",names = ["user", "item","score","timestamp"])
train_small = train_small[train_small['score'] != "rating"]
train_small.score = train_small.score.apply(float)

n_factor = 40
gamma = 0.001
lam = 0
n_iterations = 1000


conn = sqlite3.connect('comp3208_example.db')
c = conn.cursor()

n_users = train_small['user'].max()
n_items = train_small['item'].max()

p = np.random.random((n_users,n_factor))
q = np.random.random((n_items, n_factor))


for i in range(0,n_iterations):

    for _,row in tqdm(train_small.sample(frac=1).iterrows()):
        u,i,r,_ = row
        user = int(u)-1
        item = int(i)-1

        predicted_rating = np.dot(p[user],q[item])


        error = int(r) - predicted_rating
        #print(gamma * (error * q[item, :] - lam * p[user, :]))
        p[user, :] += gamma * (error * q[item, :] - lam * p[user, :])
        #print(gamma * (error * p[user, :] - lam * q[item, :]))
        q[item, :] += gamma * (error * p[user, :] - lam * q[item, :])
    print(p)
    print(q)
"""




