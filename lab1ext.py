import gensim
import nltk
from nltk.corpus import brown
import string
import scipy.sparse
import numpy as np
from scipy import spatial
from scipy.sparse import coo_matrix, dok_matrix, find
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.sparse.linalg import svds   
import math
import os
import logging
from lab1 import *
from numpy import linalg as LA



def get_most_similar(vector, matrix, word_dict):
    max_sim = -float('Inf')
    most_similar_word = ""
    for key, value in word_dict.items():
        cosine_sim = sklearn.metrics.pairwise.cosine_similarity(vector, matrix.getrow(value))
        if cosine_sim > max_sim:
            max_sim = cosine_sim
            most_similar_word = key
    return most_similar_word

# def cos_sim_val(row, vector):
#     ans = 1 - scipy.spatial.distance.cosine(vector, row)
#     print(ans)
#     return ans



def lsa_accuracy(filename, W, matrix, word_dict):
    question_words = open("./questions-words.txt")
    quest_list = question_words.readlines()
    question_words.close()

    category_acc = []
    acc_sum= 0
    cat_length = 0
    result_list = []
    print("quest_list: ")
    print(quest_list)
    #sparse_matrix = csr_matrix(np.asmatrix(matrix))
    #inv_word_dict= {v: k for k, v in word_dict.items()}

    length = len(quest_list)
    for i in range(length):
        if quest_list[i][0] == ':':
            # if i == 0:
            #     category_acc.append(quest_list[i].strip())
            #     continue
            category_acc.append(quest_list[i].strip())
            #acc_sum = 0
            #cat_length = 0
            #num_true = 0
            #num_false = 0
        else:
            #cat_length +=1
            curr = quest_list[i].split()
            print("curr: ")
            one = curr[0].lower()
            two = curr[1].lower()
            three = curr[2].lower()
            four = curr[3].lower()
            print(curr)
            if (one not in word_dict or two not in word_dict) or (three not in word_dict or four not in word_dict):
                continue
            else:
                #~~~~~~~~ let x be the third word, i.e. Berlin(v1):Germany(v2)::x(v3):France(v4)
                v1 = matrix.getrow(word_dict[one])
                v2 = matrix.getrow(word_dict[two])
                x = matrix.getrow(word_dict[three])
                v4 = matrix.getrow(word_dict[four])
                diff = np.subtract(v1.astype(np.float_), v2.astype(np.float_))
                answer = np.add(diff, v4.astype(np.float_))
                print("three: %s" % three)
                predicted = get_most_similar(answer, matrix, word_dict)
                #all_cosines = np.apply_along_axis(cos_sim_val, 1, matrix, x)
                #predicted = inv_word_dict[argmax(all_cosines)]
                print("predicted: %s" % predicted)
                #temp = sklearn.metrics.pairwise.cosine_similarity(x, answer)
                if three == predicted:
                    category_acc.append(True)
                else:
                    #num_false += 1
                    category_acc.append(False)
                #sim = temp[0][0]
                #print(sim)
                #category_acc.append(sim)

    length = len(category_acc)
    num_true = 0
    num_false = 0
    print(category_acc)
    for i in range(length):
        print(category_acc[i])
        if type(category_acc[i]) == str:
            if i == 0:
                result_list.append(category_acc[i])
                continue
            if type(category_acc[i+1]) == str:
                continue
            result_list.append(num_true/(num_true + num_false))
            result_list.append(category_acc[i])

            num_true = 0
            num_false = 0
            #acc_sum = 0
            #cat_length = 0
        else:
            if category_acc[i] == True:
                num_true += 1
            else:
                num_false +=1
            #acc_sum += category_acc[i]
            #cat_length += 1
            if i == length-1:
                result_list.append(num_true/(num_true + num_false))
                #result_list.append(acc_sum/cat_length)



    with open("LSA_analogies_lab1ext.txt", 'w') as f:
        avg_sum = 0
        avg_count = 0

        for i in range(0, len(result_list),2):
            if result_list[i][0] == ':':
                avg_count += 1
                f.write("Category %s: %.2f" % (result_list[i], result_list[i+1] + "\n"))
        f.write("\nTotal Averaged LSA accuracy: %.2f" % (avg_sum/avg_count))


if __name__ == "__main__":
    #~~~~~~~~~~~log the analogy test results to a file
    logging.basicConfig(filename="analogy_test_word2vec.log", format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    """ f = open('../../word2vec_pretrain_vec/freebase-vectors-skipgram1000.bin', 'rb')
    data = f.read()
    f.close()
    data = data[0:1000]
    data = data.decode("utf-8", 'backslashreplace')
    print(data)"""
    f = open('../RG65.txt', 'r')



    RG65 = []
    RG = [line.strip() for line in f.readlines()]
    f.close()
    for item in RG:
        bigram_list = item.split(',')
        bigram_tuple = ((bigram_list[0], bigram_list[1]), float(bigram_list[2]))
        #print(bigram_tuple)
        #print(bigram_list)
        RG65.append(bigram_tuple)

    print(RG65)
    #print(RG)
    punct = string.punctuation +  u"“‘’--ʺ”"
    corpus = [i for i in brown.words() if (i not in punct)]
    writer = open("Brown_5000_+n.txt", 'w')
    W = create_W(corpus, RG65)
    zero_count = [i for i in W if i[1] == 0]
    print(zero_count)
    #print(W)
    W_freq = [w[1] for w in W]
    W_token_size = 0
    for count in W_freq:
        W_token_size += count
    print(W_token_size)

    #for item in W:
    #    writer.write(item[0] + ',' + str(item[1]))
    #writer.close()
    #model = gensim.models.KeyedVectors.load_word2vec_format('../../word2vec_pretrain_vec/GoogleNews-vectors-negative300.bin', binary=True)
    #print(model["hello"])

    #~~~~~~~~~~Extract pre-trained word embeddings~~~~~~
    # sims = []
    # for item in RG65:
    #     word1 = item[0][0]
    #     word2 = item[0][1]
    #     sims.append(model.similarity(word1, word2))
    # sims = np.asarray(sims)

    #human_sims = np.asarray([i[1] for i in RG65])
    #print("The pearson correlation between word2vec and human similarity is: ")
    #print(pearsonr(sims, human_sims))

    #~~~~~~~~~~Perform analogy test with pre-trained word embeddings
    #model.accuracy("./questions-words.txt")

    #~~~~~~~~~~Look at accuracy when word2vec has been trained with the same words as LSA
    #model =
    #~~~~~~~~~~Matrix Reformation from Lab1~~~~~~~~~~~~~
    W_set = [w[0] for w in W]
    bigrams = nltk.bigrams(corpus)

    bigram_freq = nltk.FreqDist(bigrams).most_common()
    length = len(bigram_freq)
    top_bigram_freq = [i for i in bigram_freq if (i[0][0] in W_set and i[0][1] in W_set)]
    top_bigram_count = [i[1] for i in top_bigram_freq]
    word_dict = create_unigramdict(W)
    print("testing for they them me my")
    print(word_dict['they'])
    print(word_dict['their'])
    print(word_dict['i'])
    print(word_dict['my'])

    #print(word_dict)

    #make M1
    data  = create_row_col(top_bigram_freq, W, word_dict)
    coo = coo_matrix(data)
    #print(data)

    total_bigram = 0
    for count in top_bigram_count:
        total_bigram += count

    serf_index = word_dict['serf']
    serf_row = data.getrow(serf_index)
    serf_col = data.getcol(serf_index)


    #print(serf_row)
    #print(serf_col)

    index_dict = create_indexdict(W)
    pmi_matrix = ppmi(index_dict, data, total_bigram, W_token_size)

    serf_row_pmi = pmi_matrix.getrow(serf_index)
    serf_col_pmi = pmi_matrix.getcol(serf_index)

    #print(serf_row_pmi)
    #print(serf_col_pmi)


    u_100, v_100, d_100 = scipy.sparse.linalg.svds(pmi_matrix, 100)
    #u_50, v_50, d_50 = scipy.sparse.linalg.svds(pmi_matrix, 50)
    #u_10, v_10, d_10 = scipy.sparse.linalg.svds(pmi_matrix, 10)

    # used data instead of coo here
    r_m1 = "r value for M1: " + str(compare_similarity(RG65, data, word_dict)) + "\n"
    r_m1plus = "r value for M1+: " + str(compare_similarity(RG65, pmi_matrix, word_dict)) + "\n"

    #r_m2_10 = "r value for M2_10: " + str(compare_similarity(RG65, dok_matrix(np.asmatrix(u_10)), word_dict)) + "\n"
    #r_m2_50 = "r value for M2_50: " + str(compare_similarity(RG65, dok_matrix(np.asmatrix(u_50)), word_dict)) + "\n"
    r_m2_100 = "r value for M2_100: " + str(compare_similarity(RG65, dok_matrix(np.asmatrix(u_100)), word_dict)) + "\n"
    print(u_100.shape)
    lsa_accuracy("questions-words.txt", W, dok_matrix(np.asmatrix(u_100)), word_dict)

    # with open('pearson_coefficients_results_lab1ext', 'w') as f:
    #     f.write('where r is the pearson coefficient:\n')
    #     f.write(r_m1)
    #     f.write(r_m1plus)
    #     f.write(r_m2_10)
    #     f.write(r_m2_50)
    #     f.write(r_m2_100)
    #     f.close()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






