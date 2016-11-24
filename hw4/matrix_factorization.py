import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from nltk.corpus import reuters
from sklearn.feature_extraction import text
get_ipython().magic('matplotlib inline')

# Data import curtesy Conner
# How it works: For each file id in the nltk reuters corpus,
# get the words in that file and separate them by spaces and 
# make them lower case. This yields an array of text documents
X = np.array([" ".join(list(reuters.words(file_id))).lower()
              for file_id in reuters.fileids()])

# Convert to sparse matrix of s where a row is a document's frequency vector
tfidf = text.TfidfVectorizer()
X = tfidf.fit_transform(X)

# get labels
labels = np.array(tfidf.get_feature_names())

def objective(V, W, h):
    """
    Frobenius norm of V - WH
    """
    return np.linalg.norm(v - w @ h, ord = 'fro')

def matrix_decompose(V, k, iters):
    W = np.abs(np.random.randn(V.shape[0], k))
    H = np.abs(np.random.randn(k, V.shape[1]))
    
    objectives = [objective(V, W, H)]
    for i in range(iters):
        Wc = W.copy()
        W *= (V @ H.T) / (W @ (H @ H.T))
        H *= (Wc.T @ V) / (Wc.T @ Wc @ H)
              
        obj = objective(V, W, H)
        if i % (iters/5) == 0:
            print(obj)
        objectives.append(obj)
        
    return W, H, objectives


# Decompose. My computer could not hang, so I did 5 iterations

W,H,objs = matrix_decompose(X, 20, 5)

# plot objective
plt.plot(objs)

# Use argpartion for O(n) max args
top_words = np.array(tfidf.get_feature_names())[
            np.argpartition(H, axis=1, kth = -5)]

print(top_words[:,-5:])



