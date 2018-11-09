# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
data = dataset.values
X = data[:,1:]
Y = data[:,0]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state = 1)

def draw_img(X):
    img = X.reshape(28,28)
    plt.imshow(img,cmap='gray')
    plt.show()

def dist(X,Y):
    return np.sqrt(sum((X-Y)**2))

def KNN(X,Y,query_point,k=5):
    
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(query_point,X[i])
        vals.append((d,Y[i]))
    
    vals = sorted(vals)
    
    vals = vals[:k]
    vals = np.array(vals)
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]

    return pred
    
for i in range(10):
    print(KNN(X_train,Y_train,X_test[i]),end=' ')
    print(Y_test[i])
    draw_img(X_test[i])