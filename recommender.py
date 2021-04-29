import sqlite3

import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
np.random.seed(10)
from keras.models import Model
from keras.layers import Input, Reshape, Dot
from keras.layers import Add, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l2

'''
class RecommenderSystem:

    def __init__(self, k=40, gamma=0.001, lam=0, db=sqlite3.connect('example_table.db')):
        self.k = k
        self.gamma = gamma
        self.lam = lam
        self.db = db.cursor()
        self.item_set = {}
        self.user_set = {}

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
        for i in tqdm(range(iterations)):
            for row in self.get_shuffled_rows():
                u, i, r, _ = row
                u = u - 1
                i = i - 1
                self.user_set.add(u)
                self.item_set.add(i)

                predicted_rating = np.dot(self.p[u], self.q[i])

                error = r - predicted_rating
                self.p[u, :] += self.gamma * (error * self.q[i, :] - self.lam * self.p[u, :])
                self.q[i, :] += self.gamma * (error * self.p[u, :] - self.lam * self.q[i, :])


    def predict(self, user, item):
        if user not in self.user_set or item not in self.item_set:
            prediction = self.db.execute('SELECT AVG(Rating) FROM example_table')
        else:
            prediction =  self.p[user, :].dot(self.q[item, :].T)
        self.db.execute('UPDATE example_table SET PredRating=? WHERE UserID=? AND WHERE ItemID=?',(prediction,user,item))
        return prediction


recommender = RecommenderSystem()
'''


"""
train_small = pandas.read_csv(r"comp3208-train-small.csv",names = ["user", "item","score","timestamp"])
train_small = train_small[train_small['score'] != "rating"]
train_small.score = train_small.score.apply(float)

n_factor = 40
gamma = 0.001
lam = 0
n_iterations = 1000


conn = sqlite3.connect('example_table.db')
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

class RecommenderSystem:

    def __init__(self,n_items,n_factors):
        self.n_items = n_items
        self.n_factors = n_factors

    def __call__(self,x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,)) (x)
        return x

def recommender_system(n_users,n_items,n_factors,min_rating,max_rating):
    user = Input(shape=(1,))
    u = RecommenderSystem(n_users,n_factors)(user)
    ux = RecommenderSystem(n_users,1)(user)

    item = Input(shape=(1,))
    i = RecommenderSystem(n_items, n_factors)(item)
    ix = RecommenderSystem(n_items, 1)(item)

    x = Dot(axes=1)([u,i])
    x = Add()([x,ux,ix])
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

    model = Model(inputs=[user,item],outputs=x)
    opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error',optimizer = opt)

    return model



def process_data(data):
    data = data[data['score'] != "rating"]

    # encode user and movie fields to sequential integers
    encoder = LabelEncoder()
    data["user"] = encoder.fit_transform(data["userId"].values)
    users = data["user"].nunique()

    data["item"] = encoder.fit_transform(data["itemId"].values)
    items = data["item"].nunique()

    data["score"] = data["score"].values.astype(np.float32)
    min_rating = min(data["score"])
    max_rating = max(data["score"])

    # create list of ((user,item),score) pairs
    x = data[["user","item"]].values
    y = data["score"].values

    x = [x[:,0],x[:,1]]

    return x,y,users,items,min_rating,max_rating


train_small = pandas.read_csv(r"comp3208-train.csv",names = ["userId", "itemId","score","timestamp"])
x_train, y_train,users,items,min_rating,max_rating = process_data(train_small)

test_small = pandas.read_csv(r"comp3208-test.csv",names = ["userId", "itemId","score","timestamp"])
x_test, y_test,_,_,_,_ = process_data(test_small)

model = recommender_system(users,items,50,min_rating,max_rating)
model.summary()

history = model.fit(x=x_train, y=y_train, batch_size=64000, epochs=5, verbose=1, validation_data=(x_test,y_test))

model.save_weights("model.hdf5")

results = model.predict(x_test)
print(results)