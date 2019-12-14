from sklearn import svm
import os
import random
import pickle
from collections import Counter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


"""
    This code tests the best performing classifers on new unseen review data collected from IMDB
    The doc2vec models used to run this code for the results found in my writeup, along with the saved best models for
    each setup can be downloaded here: 
    https://drive.google.com/file/d/1wCk_b-efz4r9xBGX7FD4NqT0x9-usYXi/view?usp=sharing    
    They are also available in my area on the MPhil machines (jw2088)
"""

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


def vectorise(rev, vocab_dict, n=1, freq=True):
    # Takes tokenised review in text form and encodes it as a Bag of Words vector.
    # If freq is true, the ith element shows the frequency of word w_i, if freq is False it indicates its presence with
    # a 1 or 0.
    vec = [0] * len(vocab_dict)
    for i in range(len(rev) - n + 1):
        w = " ".join([rev[i + j] for j in range(n)])
        if w not in vocab_dict:
            w = " ".join(["UNK"] * n)
        if freq:
            vec[vocab_dict[w]] += 1
        else:
            vec[vocab_dict[w]] = 1
    return vec


def examine_d2v_class(kernel, model):
    # Given a particular d2v model this allows to see all incorrectly classified reviews and their classifications
    pos_file_list = [os.path.join("new_test_reviews", "pos", f)
                     for f in os.listdir(os.path.join("new_test_reviews", "pos"))]
    neg_file_list = [os.path.join("new_test_reviews", "neg", f)
                     for f in os.listdir(os.path.join("new_test_reviews", "neg"))]
    pos, neg = read_files(pos_file_list, neg_file_list)
    svm = pickle.load(open("best_model_d2v.sav", "rb"))[kernel][model]
    d2v_model = Doc2Vec.load(model)
    for p in pos:
        prediction = svm.predict([d2v_model.infer_vector(p)])[0]
        if prediction != 1:
            print("Positive review misclassified:")
            print(p)
    for n in neg:
        prediction = svm.predict([d2v_model.infer_vector(n)])[0]
        if prediction != 0:
            print("Negative review misclassified:")
            print(n)


def test_all_new_reviews_d2v():
    # Accuracy tested on all of the new reviews for d2v models
    pos_file_list = [os.path.join("new_test_reviews", "pos", f)
                     for f in os.listdir(os.path.join("new_test_reviews", "pos"))]
    neg_file_list = [os.path.join("new_test_reviews", "neg", f)
                     for f in os.listdir(os.path.join("new_test_reviews", "neg"))]
    pos, neg = read_files(pos_file_list, neg_file_list)
    saved_models = pickle.load(open("best_model_d2v.sav", "rb"))
    kernels = ["rbf", "linear", "poly", "sigmoid"]
    model_list = {"dbow.model", "dm.model", "dbow100.model", "dm100.model", "dbowlarge.model", "dmlarge.model",
                  "dbow5cutoff.model", "dm5cutoff.model", "dbow1cutoff.model", "dm1cutoff.model"}
    for kernel in kernels:
        for model in model_list:
            pos_corr = 0
            neg_corr = 0
            pos_tot = len(pos)
            neg_tot = len(neg)
            svm = saved_models[kernel][model]
            d2v_model = Doc2Vec.load(model)
            for p in pos:
                prediction = svm.predict([d2v_model.infer_vector(p)])[0]
                if prediction == 1:
                    pos_corr += 1
            for n in neg:
                prediction = svm.predict([d2v_model.infer_vector(n)])[0]
                if prediction == 0:
                    neg_corr += 1
            print(kernel, model, ", Accuracy: ", (pos_corr+neg_corr)/(pos_tot+neg_tot))


def test_all_new_reviews_non_d2v():
    # Accuracy tested on all of the new reviews for BoW models
    param_sets = {"unifreq": {"n": 1, "freq": True, "vocab_dict": "uni", "kernel": "rbf"},
                  "unipres": {"n": 1, "freq": False, "vocab_dict": "uni", "kernel": "rbf"},
                  "bifreq": {"n": 2, "freq": True, "vocab_dict": "bi", "kernel": "rbf"},
                  "bipres": {"n": 2, "freq": False, "vocab_dict": "bi", "kernel": "rbf"},
                  "unifreqlin": {"n": 1, "freq": True, "vocab_dict": "uni", "kernel": "linear"},
                  "unipreslin": {"n": 1, "freq": False, "vocab_dict": "uni", "kernel": "linear"},
                  "bifreqlin": {"n": 2, "freq": True, "vocab_dict": "bi", "kernel": "linear"},
                  "bipreslin": {"n": 2, "freq": False, "vocab_dict": "bi", "kernel": "linear"},
                  "unifreqpoly": {"n": 1, "freq": True, "vocab_dict": "uni", "kernel": "poly"},
                  "uniprespoly": {"n": 1, "freq": False, "vocab_dict": "uni", "kernel": "poly"},
                  "bifreqpoly": {"n": 2, "freq": True, "vocab_dict": "bi", "kernel": "poly"},
                  "biprespoly": {"n": 2, "freq": False, "vocab_dict": "bi", "kernel": "poly"},
                  "unifreqsig": {"n": 1, "freq": True, "vocab_dict": "uni", "kernel": "sigmoid"},
                  "unipressig": {"n": 1, "freq": False, "vocab_dict": "uni", "kernel": "sigmoid"},
                  "bifreqsig": {"n": 2, "freq": True, "vocab_dict": "bi", "kernel": "sigmoid"},
                  "bipressig": {"n": 2, "freq": False, "vocab_dict": "bi", "kernel": "sigmoid"}
                  }
    pos_file_list = [os.path.join("new_test_reviews", "pos", f)
                     for f in os.listdir(os.path.join("new_test_reviews", "pos"))]
    neg_file_list = [os.path.join("new_test_reviews", "neg", f)
                     for f in os.listdir(os.path.join("new_test_reviews", "neg"))]
    pos, neg = read_files(pos_file_list, neg_file_list)
    best_classifier_params = pickle.load(open("best_model.sav", "rb"))
    for model_name in param_sets.keys():
        svm_model = best_classifier_params[model_name]["model"]
        vocab = best_classifier_params[model_name]["vocab"]
        pos_corr = 0
        pos_tot = len(pos)
        neg_corr = 0
        neg_tot = len(neg)
        params = param_sets[model_name]
        p_predictions = svm_model.predict([vectorise(p, vocab_dict=vocab, n=params["n"],
                                                     freq=params["freq"]) for p in pos])
        for i in range(len(p_predictions)):
            p = p_predictions[i]
            if p == 1:
                pos_corr += 1
        n_predictions = svm_model.predict([vectorise(ng, vocab_dict=vocab, n=params["n"],
                                                     freq=params["freq"]) for ng in neg])
        for i in range(len(n_predictions)):
            ng = n_predictions[i]
            if ng == 0:
                neg_corr += 1
        accuracy = (neg_corr + pos_corr) / (neg_tot + pos_tot)
        print(model_name, ", Accuracy: ", accuracy)


test_all_new_reviews_d2v()
test_all_new_reviews_non_d2v()