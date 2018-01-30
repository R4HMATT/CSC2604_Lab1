import nltk
from nltk.corpus import brown
import string
import scipy.sparse
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import coo_matrix, dok_matrix, find
from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr
import math
def create_W(corpus):
    """punct = punct = string.punctuation +  u"“‘’--ʺ”"
    corpus = [i for i in corpus if (i not in punc)]

    Returns the 5000 most frequent words in the corpus and their count.
    """

    frequencies = nltk.FreqDist(i.lower() for i in corpus)
    W = frequencies.most_common(5000)
    #print(W[:50])
    return W

def construct_bigram(book):
    """book is our list W of 5000 words """
    ngram = []
    for i in range(1,len(book)-1):
        ngram.append(book[i:i+2])
    return ngram


def create_indexdict(unigram):
    # creates a dictionary where the keys are our supposed ordering of the words in the 5000x5000 matrix, and the values are the actual words the index represents
    index_dict = {}
    for i in range(len(unigram)):
        index_dict[i] = unigram[i][1]
    return index_dict

def create_unigramdict(unigrm):
    # creates a dictionary where the keys are the words in the unigram, and the value is its index in the matrix
    word_dict = {}

    for i in range(len(unigrm)):
        word_dict[unigrm[i][0]] = i
    return word_dict

def create_row_col(bgrm, unigrm, word_dict):
    """this function takes in a list bigrm where each element contains a tuple of a bigram, and its corresponding freq count
    in W. the first element in the tuple is the word, and the second is its context."""
    """Premise of this function is that the rows and cols of the matrix we create will be in order of the list of unigram frequencies (just for conveniences sake) """
    """unigrm is the set of all token types listed in order of frequencies (decreasing order)"""
    """word_dict lets us look up the words in the bigram, to find their index number which we use as the row and column indices"""
    # word
    row = []
    # context
    col = []
    row_index = []
    col_index = []
    i=0
    j=0
    data =[[0]*5000]*5000

    # use a sparse matrix for efficiency
    sparse = dok_matrix((5000,5000))
    for item in bgrm:
        row_index = word_dict[item[0][0]]
        col_index = word_dict[item[0][1]]
        sparse[row_index, col_index] = item[1]

    return sparse

def ppmi(uni_dict, matrix, total_bigram, total_unigram):
     """ calculates ppmi matrix of the given matrix."""
     """uni_dict has keys as the row/col indices of the W frequency distribution, and the value is that word's total unigram frequency, total_bigram is how many bigram tokens there are, total_unigram is how many unigrams there are"""

     # find method gets all the non-zero values of our sparse matrix
     values = find(matrix)
     print(values)
     num_values = values[0].size
     for i in range(num_values):
         row = values[0].item(i)
         col = values[1].item(i)
         prow = uni_dict[row]/total_unigram
         pcol = uni_dict[col]/total_unigram
         matrix[row,col] = max(0, math.log((matrix[row,col]/total_unigram)/(prow * pcol),  2 ))

     return matrix

def cosine_similarity(word1, word2, matrix, word_dict):
    row1, row2 = word_dict[word1], word_dict[word2]
    v1, v2 = matrix.getrow(row1).toarray(), matrix.getrow(row2).toarray()
    #dist = 1 - (np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    sim = sklearn.metrics.pairwise.cosine_similarity(v1, v2)[0][0]
    return sim

def compare_similarity(judged, matrix, word_dict):
    judged_sims = [i[1] for i in judged]
    cosine_sims = []
    for item in judged:
        cosine_sims.append(cosine_similarity(item[0][0], item[0][1], matrix, word_dict))
    corr, p_val = pearsonr(judged_sims, cosine_sims)
    return corr


if __name__ == "__main__":
    punct = punct = string.punctuation +  u"“‘’--ʺ”"

    #~~~~~~~~~~~~~~JUDGED SIMILARITY BETWEEN ITEMS~~~~~~~~~~~~~~~~~
    judged_sim = [(("cord", "smile"), 0.02), (("rooster", "voyage"), 0.04), (("noon", "string"), 0.04), (("fruit", "furnace"), 0.05), (("autograph", "shore"), 0.06), (("automobile", "wizard"), 0.11),
            (("mound", "stove"), 0.14), (("grin", "implement"), 0.18), (("asylum", "fruit"), 0.19), (("asylum", "monk"), 0.39), (("graveyard", "madhouse"), 0.42), (("glass", "magician"), 0.44),
            (("boy", "rooster"), 0.44), (("cushion", "jewel"), 0.45), (("monk", "slave"), 0.57), (("asylum", "cemetary"), 0.79), (("coast", "forest"), 0.85), (("grin", "lad"), 0.88), (("shore", "woodland"), 0.9),
            (("monk", "oracle"), 0.91), (("boy", "sage"), 0.96), (("automobile", "cushion"), 0.97), (("mound", "shore"), 0.97), (("lad", "wizard"), 0.99), (("forest", "graveyard"), 1),
            (("food", "rooster"), 1.09), (("cemetery", "woodland"), 1.18), (("shore", "voyage"), 1.22), (("bird", "woodland"), 1.24), (("coast", "hill"), 1.26), (("furnace", "implement"), 1.37),
            (("crane", "rooster"), 1.41), (('hill', 'woodland'), 1.48), (('car', 'journey'), 1.55), (('cemetery', 'mound'), 1.69), (('glass', 'jewel'), 1.78), (('magician','oracle'), 1.82), (('crane', 'implement'), 2.37), (('brother', 'lad'), 2.41), (('sage', 'wizard'), 2.46), (('oracle', 'sage'), 2.61), (('bird', 'crane'), 2.63), (('bird', 'cock'), 2.63), (('food', 'fruit'), 2.69), (('brother', 'monk'), 2.74), (('asylum', 'madhouse'), 3.04),(('furnace', 'stove'), 3.11), (('magician', 'wizard'),3.21),(('hill', 'mound'), 3.29), (('cord', 'string'), 3.41), (('glass', 'tumbler'), 3.45), (('grin', 'smile'), 3.46), (('serf', 'slave'), 3.46), (('journey', 'voyage'), 3.58), (('autograph', 'signature'), 3.59), (('coast', 'shore'), 3.60), (('forest', 'woodland'), 3.65), (('implement', 'tool'), 3.66), (('cock', 'rooster'), 3.68), (('boy', 'lad'), 3.82), (('cushion', 'pillow'), 3.84), (('cemetery', 'graveyard'), 3.88), (('automobile', 'car'), 3.92), (('midday', 'noon'), 3.94), (('gem', 'jewel'), 3.94)]

    #set everything up
    corpus = [i for i in brown.words() if (i not in punct)]
    W = create_W(corpus)
    W_set = [w[0] for w in W]
    W_freq = [w[1] for w in W]
    top_judged_sim = [w for w in judged_sim if (w[0][0] in W_set and w[0][1] in W_set)]
    bigrams = nltk.bigrams(corpus)
    bigram_freq = nltk.FreqDist(bigrams).most_common()
    length = len(bigram_freq)
    top_bigram_freq = [i for i in bigram_freq if (i[0][0] in W_set and i[0][1] in W_set)]
    top_bigram_count = [i[1] for i in top_bigram_freq]
    word_dict = create_unigramdict(W)

    #make M1
    data  = create_row_col(top_bigram_freq, W, word_dict)
    coo = coo_matrix(data)

    total_bigram = 0
    for count in top_bigram_count:
        total_bigram += count

    W_token_size = 0
    for count in W_freq:
        W_token_size += count

    index_dict = create_indexdict(W)

    #make M1+
    pmi_matrix = ppmi(index_dict, data, total_bigram, W_token_size)

    #make M2_10, M2_50, M2_100
    svd_10 = TruncatedSVD(10, "arpack")
    svd_50 = TruncatedSVD(50, "arpack")
    svd_100 = TruncatedSVD(100, "arpack")
    ten_dim = coo_matrix(svd_10.fit_transform(pmi_matrix))
    fifty_dim = coo_matrix(svd_50.fit_transform(pmi_matrix))
    hund_dim = coo_matrix(svd_100.fit_transform(pmi_matrix))


    print("r value for M1: " + str(compare_similarity(top_judged_sim, coo, word_dict)))
    print("r value for M1+: " + str(compare_similarity(top_judged_sim, pmi_matrix, word_dict)))
    print("r value for M2_10: " + str(compare_similarity(top_judged_sim, ten_dim, word_dict)))
    print("r value for M2_50: " + str(compare_similarity(top_judged_sim, fifty_dim, word_dict)))
    print ("r value for M2_100: " + str(compare_similarity(top_judged_sim, hund_dim, word_dict)))




