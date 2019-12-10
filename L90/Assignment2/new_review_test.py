from sklearn import svm
import os
import random
import pickle
from collections import Counter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def read_files(pos, neg):
    # Given a list of positive file names and negative file names, reads in the text data from each file.
    pos_text = []
    neg_text = []
    for pos_f in pos:
        with open(pos_f, 'r') as f:
            (pos_text.append(f.read().split(" ")))
    for neg_f in neg:
        with open(neg_f, 'r') as f:
            neg_text.append(f.read().split(" "))
    return pos_text, neg_text


def test_new_reviews():
    pos_file_list = [os.path.join("new_test_reviews", "pos", f)
                     for f in os.listdir(os.path.join("new_test_reviews", "pos"))]
    neg_file_list = [os.path.join("new_test_reviews", "neg", f)
                     for f in os.listdir(os.path.join("new_test_reviews", "neg"))]
    pos, neg = read_files(pos_file_list, neg_file_list)
    svm = pickle.load(open("best_model.sav", "rb"))
    d2v_model = Doc2Vec.load("dbowlarge.model")
    for p in pos:
        if svm.predict(d2v_model.infer_vector(p)) != 1:
            print(p)
    for n in neg:
        if svm.predict(d2v_model.infer_vector(n)) != 1:
            print(n)


test_new_reviews()