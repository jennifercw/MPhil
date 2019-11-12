from sklearn import svm
import os
import random
from collections import Counter
from nltk import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

"""
    The models used to run this code for the results found in my writeup can be downloaded here: 
    https://drive.google.com/file/d/1ykdC4qwMj-nZC6t5P5KJQZespIaH__7-/view?usp=sharing
"""

# TODO: Unigram + bigram model
# TODO: Test best on blind test set
# TODO: Do significance testing

def read_round_robin_groups(k=10):
    # Returns k lists of file names, divided up round robin style.
    pos_file_list = [os.path.join("data", "POS", f) for f in os.listdir(os.path.join("data", "POS"))]
    neg_file_list = [os.path.join("data", "NEG", f) for f in os.listdir(os.path.join("data", "NEG"))]
    assert (len(pos_file_list) == len(neg_file_list))
    pos_round_robins = []
    neg_round_robins = []
    for i in range(k):
        pos_round_robins.append([])
        neg_round_robins.append([])
    num_files = len(pos_file_list)
    i = 0
    j = 0
    while i < num_files:
        pos_round_robins[j].append(pos_file_list[i])
        neg_round_robins[j].append(neg_file_list[i])
        i += 1
        j += 1
        if j == k:
            j = 0
    return pos_round_robins, neg_round_robins


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


def build_vocab(pos_files, neg_files, n=1):
    # Counts how many times every word appears in the files and determines an appropriate cutoff for the vocabulary
    # size based on word frequency.
    # Creates a mapping from vocab words to integers and a reverse mapping.
    all_files = pos_files + neg_files
    all_vocab = Counter()
    for f in all_files:
        for i in range(len(f) - n + 1):
            w = " ".join([f[i + j] for j in range(n)])
            all_vocab[w] += 1
    # Cutoff decided through experimentation as to what provided a reasonable vocab size.
    cutoff = sum([1 for i in all_vocab.items() if i[1] >= 5])
    top_vocab = [v[0] for v in all_vocab.most_common(cutoff)]
    # Unknown token to capture anything that falls outside of the vocabulary.
    top_vocab.append(" ".join(["UNK"] * n))
    vocab_dict = {top_vocab[i]: i for i in range(len(top_vocab))}
    reverse_dict = {c: v for v, c in vocab_dict.items()}
    return vocab_dict, reverse_dict


def vectorise(rev, vocab_dict, n=1, freq=True):
    # Takes tokenised review in text form and encodes it as a Bag of Words vector.
    # If freq is true, the ith element shows the frequency of word w_i, if freq is False it indicates its presence with
    # a 1 or 0.
    vec = [0] * len(vocab_dict)
    for i in range(len(rev) - n + 1):
        w = " ".join([rev[i + j] for j in range(n)])
        if w not in vocab_dict:
            w = " ".join(["UNK"] * n)
        if freq == True:
            vec[vocab_dict[w]] += 1
        else:
            vec[vocab_dict[w]] = 1
    return vec

def run_test(k=10):
    model_list = ["dbow.model", "dm.model", "dbow100.model", "dm100.model", "dbowlarge.model", "dmlarge.model",
                  "dbow5cutoff.model", "dm5cutoff.model", "dbow1cutoff.model", "dm1cutoff.model"]
    param_names = ["unifreq", "unipres", "bifreq", "bipres"]
    acc_vals = {name: [] for name in param_names}
    best_classifier_params = {name : None for name in param_names}
    best_classifier_doc2vec = {model_name : None for model_name in model_list}
    pos_robins, neg_robins = read_round_robin_groups(k)
    pos_test = pos_robins[-1]
    neg_test = neg_robins[-1]
    pos_test_data, neg_test_data = read_files(pos_test, neg_test)
    pos_robins = pos_robins[:-1]
    neg_robins = neg_robins[:-1]
    acc_list = {model_name : [] for model_name in model_list}

    for i in range(k - 1):
        print(i)
        val_pos = pos_robins[i]
        val_neg = neg_robins[i]
        pos_val_data, neg_val_data = read_files(val_pos, val_neg)
        train_pos = []
        train_neg = []
        for j in range(k - 1):
            if j == i:
                continue
            else:
                train_neg.extend(neg_robins[j])
                train_pos.extend(pos_robins[j])
        train_pos_data, train_neg_data = read_files(train_pos, train_neg)
        uni_vocab_dict, uni_reverse_dict = build_vocab(train_pos_data, train_neg_data, n=1)
        bi_vocab_dict, bi_reverse_dict = build_vocab(train_pos_data, train_neg_data, n=2)
        param_sets = {"unifreq": {"n": 1, "freq": True, "vocab_dict": uni_vocab_dict},
                      "unipres": {"n": 1, "freq": False, "vocab_dict": uni_vocab_dict},
                      "bifreq": {"n": 2, "freq": True, "vocab_dict": bi_vocab_dict},
                      "bipres": {"n": 2, "freq": False, "vocab_dict": bi_vocab_dict}}
        for model_name in model_list:
            model = Doc2Vec.load(model_name)
            train_w_labels = [(model.infer_vector(p), 1) for p in train_pos_data] + \
                             [(model.infer_vector(ng), 0) for ng in train_neg_data]
            random.shuffle(train_w_labels)
            X = [rev[0] for rev in train_w_labels]
            Y = [rev[1] for rev in train_w_labels]
            svm_model = svm.SVC(gamma='scale', kernel='sigmoid')
            svm_model.fit(X, Y)
            pos_corr = 0
            pos_tot = len(pos_val_data)
            neg_corr = 0
            neg_tot = len(neg_val_data)
            p_predictions = svm_model.predict([model.infer_vector(p) for p in pos_val_data])
            for p in p_predictions:
                if p == 1:
                    pos_corr += 1
            n_predictions = svm_model.predict([model.infer_vector(ng)for ng in neg_val_data])
            for ng in n_predictions:
                if ng == 0:
                    neg_corr += 1
            accuracy = (neg_corr + pos_corr) / (neg_tot + pos_tot)
            print(accuracy)
            acc_list[model_name].append(accuracy)
            if accuracy == max(acc_list[model_name]):
                best_classifier_doc2vec[model_name] = svm_model


        for name, params in param_sets.items():
            print(name)
            train_w_labels = [(vectorise(p, vocab_dict=params["vocab_dict"], n=params["n"], freq=params["freq"]), 1)
                              for p in train_pos_data] + [(vectorise(ng, vocab_dict=params["vocab_dict"], n=params["n"],
                                                                     freq=params["freq"]), 0) for ng in train_neg_data]

            random.shuffle(train_w_labels)

            X = [rev[0] for rev in train_w_labels]
            Y = [rev[1] for rev in train_w_labels]
            svm_model = svm.SVC(gamma='scale', kernel='sigmoid')
            svm_model.fit(X, Y)
            pos_corr = 0
            pos_tot = len(pos_val_data)
            neg_corr = 0
            neg_tot = len(neg_val_data)
            p_predictions = svm_model.predict([vectorise(p, vocab_dict=params["vocab_dict"], n=params["n"],
                                                         freq=params["freq"]) for p in pos_val_data])
            for p in p_predictions:
                if p == 1:
                    pos_corr += 1
            n_predictions = svm_model.predict([vectorise(ng, vocab_dict=params["vocab_dict"], n=params["n"],
                                                         freq=params["freq"]) for ng in neg_val_data])
            for ng in n_predictions:
                if ng == 0:
                    neg_corr += 1
            accuracy = (neg_corr + pos_corr) / (neg_tot + pos_tot)
            print(name, accuracy)
            acc_vals[name].append(accuracy)
            if accuracy == max(acc_vals[name]):
                best_classifier_params[name] = svm_model
    for name in param_names:
         print(name, calc_mean_variance(acc_vals[name]))
    for model_name in model_list:
        print(model_name, calc_mean_variance(acc_list[model_name]))
    """for name in param_names:
        # Current problem is that we need appropriate vocab_dict as well
        # Think make the vocab dict indexed by k rather than model
        # Model can index uni or bi and thus can be defined before loop
        svm_model = best_classifier_params[name]
        pos_corr = 0
        pos_tot = len(pos_test_data)
        neg_corr = 0
        neg_tot = len(neg_test_data)
        params = param_sets[name]
        p_predictions = svm_model.predict([vectorise(p, vocab_dict=params["vocab_dict"], n=params["n"],
                                                     freq=params["freq"]) for p in pos_test_data])
        for p in p_predictions:
            if p == 1:
                pos_corr += 1
        n_predictions = svm_model.predict([vectorise(ng, vocab_dict=params["vocab_dict"], n=params["n"],
                                                     freq=params["freq"]) for ng in neg_test_data])
        for ng in n_predictions:
            if ng == 0:
                neg_corr += 1
        accuracy = (neg_corr + pos_corr) / (neg_tot + pos_tot)
        print(name, accuracy)"""


def calc_mean_variance(acc):
    # Given a list of accuracy values, calulcates their mean and variance.
    m = sum(acc) / len(acc)
    v = sum((x - m) ** 2 for x in acc) / len(acc)
    return m, v

run_test(10)