from collections import Counter
import os
import math


def read_round_robin_groups(k = 10):
    # Returns k lists of file names, divided up round robin style.
    pos_file_list = os.listdir(os.path.join("/usr", "groups", "mphil", "L90", "data", "POS"))
    neg_file_list = os.listdir(os.path.join("/usr", "groups", "mphil", "L90", "data", "NEG"))
    assert(len(pos_file_list) == len(neg_file_list))
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
        i+=1
        j+=1
        if j == k:
            j = 0
    return pos_round_robins, neg_round_robins

def read_files(pos, neg):
    # Given a list of positive file names and negative file names, reads in the text data from 
    # each file.
    pos_text = []
    neg_text = []
    for pos_f in pos:
        with open(os.path.join("/usr", "groups", "mphil", "L90","data", "POS", pos_f), 'r') as f:
            (pos_text.append(f.read().split(" ")))
    for neg_f in neg:
        with open(os.path.join("/usr", "groups", "mphil", "L90", "data", "NEG", neg_f), 'r') as f:
            neg_text.append(f.read().split(" "))
    return pos_text, neg_text


def build_vocab(pos_files, neg_files, n = 1):
    # Counts how many times every word appears in the files and determines
    # an appropriate cutoff for the vocabulary size based on word frequency.
    # Creates a mapping from vocab words to integers and a reverse mapping.
    all_files = pos_files + neg_files
    all_vocab = Counter()
    for f in all_files:
        for i in range(len(f) - n + 1):
            w = " ".join([f[i + j] for j in range(n)])
            all_vocab[w] += 1
    cutoff = sum([1 for i in all_vocab.items() if i[1] >= 5])
    top_vocab = [v[0] for v in all_vocab.most_common(cutoff)]
    # Unknown word token to capture anything that falls outside of the vocabulary.
    top_vocab.append(" ".join(["UNK"] * n))
    vocab_dict = {top_vocab[i] : i for i in range(len(top_vocab))}
    reverse_dict = {c : v for v, c in vocab_dict.items()}
    return vocab_dict, reverse_dict

def get_stats(pos_files, neg_files, vocab_dict, n = 1, smoothing_k = 1):
    # Calculates statistic for files that will allow us to determine probabilities
    # to use in our Naive Bayes model.

    # Prior probabilities for the categories.
    pos_neg_likelihood = {}
    total_pos = len(pos_files)
    total_neg = len(neg_files)
    pos_neg_likelihood["pos"] = total_pos / (total_pos + total_neg)
    pos_neg_likelihood["neg"] = total_neg / (total_pos + total_neg)

    pos_count = sum([len(rev) - n + 1 for rev in pos_files])
    neg_count = sum([len(rev) - n + 1 for rev in neg_files])
    word_counts = {"pos" : {}, "neg" : {}}
    word_probs = {"pos" : {}, "neg" : {}}

    # Laplace smoothing
    for w in vocab_dict.keys():
        word_counts["pos"][w] = smoothing_k
        word_counts["neg"][w] = smoothing_k

    # Counting every appearance of the word within positive files.    
    for p in pos_files:
        for i in range(len(p) - n + 1):
            w = " ".join([p[i + j] for j in range(n)])
            if w not in vocab_dict:
                w = " ".join(["UNK"] * n)
            word_counts["pos"][w] += 1
    # Calculating probability of word appearing in positive document.
    for w, count in word_counts["pos"].items():
        word_probs["pos"][w] = count / (pos_count + (smoothing_k * len(vocab_dict)))

    # Counting every appearance of the word within negative files.    
    for ng in neg_files:
        for i in range(len(ng) - n + 1):
            w = " ".join([ng[i + j] for j in range(n)])
            if w not in vocab_dict:
                w = " ".join(["UNK"] * n)
            word_counts["neg"][w] += 1
    # Calculating probability of word appearing in negative document.
    for w, count in word_counts["neg"].items():
        word_probs["neg"][w] = count / (neg_count + (smoothing_k * len(vocab_dict)))

    return pos_neg_likelihood, word_probs


def vectorise(rev, vocab_dict, n = 1, freq = True):
    # Takes tokenised review in text form and encodes it as a Bag of Words vector.
    # If freq is true, the ith element shows the frequency of word w_i, if freq is False
    # it indicates its presence with a 1 or 0.
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

def classify(rev_vec, class_probs, word_probs, reverse_dict):
    # Calculates the log probability of a document being positive or negative.
    # Log probability simplifies the calculations as otherwise we would be multiplying
    # very small numbers
    pos_prob = math.log(class_probs["pos"])
    neg_prob = math.log(class_probs["neg"])
    for i in range(len(rev_vec)):
        if rev_vec[i] == 0:
            continue
        else:
            if word_probs["pos"][reverse_dict[i]] == 0:
                pos_prob += -float("inf")
            else:
                pos_prob += rev_vec[i] * math.log(word_probs["pos"][reverse_dict[i]])
            if word_probs["neg"][reverse_dict[i]] == 0:
                neg_prob += -float("inf")
            else:
                neg_prob += rev_vec[i] * math.log(word_probs["neg"][reverse_dict[i]])
    if pos_prob >= neg_prob:
        return "pos"
    else:
        return "neg"

def binom_coeff(n, k):
    # Calculates binomial coefficient nCk
    return fact(n)/(fact(k) * fact(n - k))


def fact(x):
    # Calculates x!
    if x == 1 or x == 0:
        return 1
    else:
        return x * fact(x - 1)


def sign_test(acc1, acc2, q = 0.5):
    # Performs sign test to determine whether the methods used in acc1 and acc2 differ in a statistically
    # significant way.
    high = 0
    low = 0
    same = 0
    N = len(acc1)
    for i in range(N):
        if acc1[i] > acc2[i]:
            low += 1
        elif acc1[i] < acc2[i]:
            high += 1
        else:
            same += 1
    k = min(high, low)
    if same > 0:
        N = N - same
    sign_test = 2 * sum([binom_coeff(N, i) * q**i * (1 - q)**(N-i)  for i in range(k + 1)])
    return sign_test

def train_and_test_model(pos, neg, pos_test, neg_test, n = 1, smoothing_k = 1, freq = True):
    # Performs all the steps to train a model and then tests it and determines the accuracy on a held
    # out test set.
    vocab, reverse = build_vocab(pos, neg, n)
    class_p, word_p = get_stats(pos, neg, vocab, n, smoothing_k)
    pos_tot = len(pos_test)
    neg_tot = len(neg_test)
    pos_corr = 0
    neg_corr = 0
    for p in pos_test:
        vec = vectorise(p, vocab, n, freq)
        res = classify(vec, class_p, word_p, reverse)
        if res == "pos":
            pos_corr += 1
    for ng in neg_test:
        vec = vectorise(ng, vocab, n, freq)
        res = classify(vec, class_p, word_p, reverse)
        if res == "neg":
            neg_corr += 1
    acc = (neg_corr + pos_corr)/(pos_tot + neg_tot)
    return(acc)

def train_and_test_unigrams_and_bigrams(pos, neg, pos_test, neg_test, smoothing_k = 1, freq = True):
    uni_vocab, uni_reverse = build_vocab(pos, neg,1)
    bi_vocab, bi_reverse = build_vocab(pos, neg, 2)
    uni_class_p, uni_word_p = get_stats(pos, neg, uni_vocab, 1, smoothing_k)
    bi_class_p, bi_word_p = get_stats(pos, neg, bi_vocab, 2, smoothing_k)
    pos_tot = len(pos_test)
    neg_tot = len(neg_test)
    pos_corr = 0
    neg_corr = 0
    for p in pos_test:
        uni_vec = vectorise(p, uni_vocab, 1, freq)
        bi_vec = vectorise(p, bi_vocab, 2, freq)
        res = classify_uni_and_bi(uni_vec, bi_vec, uni_class_p, uni_word_p, bi_word_p, uni_reverse, bi_reverse)
        if res == "pos":
            pos_corr += 1
    for ng in neg_test:
        uni_vec = vectorise(ng, uni_vocab, 1, freq)
        bi_vec = vectorise(ng, bi_vocab, 2, freq)
        res = classify_uni_and_bi(uni_vec, bi_vec, uni_class_p, uni_word_p, bi_word_p, uni_reverse, bi_reverse)
        if res == "neg":
            neg_corr += 1
    acc = (neg_corr + pos_corr)/(pos_tot + neg_tot)
    return(acc)

def classify_uni_and_bi(uni_vec, bi_vec, uni_class_p, uni_word_p, bi_word_p, uni_reverse, bi_reverse):
    # Calculates probability of each class using both unigram and bigram vectors
    pos_prob = math.log(uni_class_p["pos"])
    neg_prob = math.log(uni_class_p["neg"])
    for i in range(len(uni_vec)):
        if uni_vec[i] == 0:
            continue
        else:
            if uni_word_p["pos"][uni_reverse[i]] == 0:
                pos_prob += -float("inf")
            else:
                pos_prob += uni_vec[i] * math.log(uni_word_p["pos"][uni_reverse[i]])
            if uni_word_p["neg"][uni_reverse[i]] == 0:
                neg_prob += -float("inf")
            else:
                neg_prob += uni_vec[i] * math.log(uni_word_p["neg"][uni_reverse[i]])
    for i in range(len(bi_vec)):
        if bi_vec[i] == 0:
            continue
        else:
            if bi_word_p["pos"][bi_reverse[i]] == 0:
                pos_prob += -float("inf")
            else:
                pos_prob += bi_vec[i] * math.log(bi_word_p["pos"][bi_reverse[i]])
            if bi_word_p["neg"][bi_reverse[i]] == 0:
                neg_prob += -float("inf")
            else:
                neg_prob += bi_vec[i] * math.log(bi_word_p["neg"][bi_reverse[i]])
    if pos_prob >= neg_prob:
        return "pos"
    else:
        return "neg"    

def calc_mean_variance(acc):
    # Given a list of accuracy values, calulcates their mean and variance.
    m = sum(acc)/len(acc)
    v = sum((x-m)**2 for x in acc)/len(acc)
    return m, v

def run_tests(k = 10):
    # RUns tests for different regimes of naive Bayes model.
    pos_robins, neg_robins = read_round_robin_groups()
    uni_no_smooth_freq = []
    uni_smooth_freq = []
    uni_no_smooth_pres = []
    uni_smooth_pres = []
    bi_no_smooth_freq = []
    bi_smooth_freq = []
    bi_no_smooth_pres = []
    bi_smooth_pres = []
    both_smooth_freq = []
    both_smooth_pres = []
    both_no_smooth_freq = []
    both_no_smooth_pres = []

    for i in range(k):
        test_set_pos = pos_robins[i]  
        test_set_neg = neg_robins[i]
        train_set_pos = []
        train_set_neg = []
        for j in range(k):
            if j == i:
                continue
            else:
                train_set_neg.extend(neg_robins[j])
                train_set_pos.extend(pos_robins[j])
        pos_train_text, neg_train_text = read_files(train_set_pos, train_set_neg)
        pos_test_text, neg_test_text = read_files(test_set_pos, test_set_neg)

        print("i = " + str(i))
        uni_no_smooth_freq.append(train_and_test_model(pos_train_text, neg_train_text, pos_test_text, neg_test_text, n = 1, smoothing_k = 0, freq = True))
        uni_smooth_freq.append(train_and_test_model(pos_train_text, neg_train_text, pos_test_text, neg_test_text, n = 1, smoothing_k = 1, freq = True))
        uni_no_smooth_pres.append(train_and_test_model(pos_train_text, neg_train_text, pos_test_text, neg_test_text, n = 1, smoothing_k = 0, freq = False))
        uni_smooth_pres.append(train_and_test_model(pos_train_text, neg_train_text, pos_test_text, neg_test_text, n = 1, smoothing_k = 1, freq = False))
        bi_no_smooth_freq.append(train_and_test_model(pos_train_text, neg_train_text, pos_test_text, neg_test_text, n = 2, smoothing_k = 0, freq = True))
        bi_smooth_freq.append(train_and_test_model(pos_train_text, neg_train_text, pos_test_text, neg_test_text, n = 2, smoothing_k = 1, freq = True))
        bi_no_smooth_pres.append(train_and_test_model(pos_train_text, neg_train_text, pos_test_text, neg_test_text, n = 2, smoothing_k = 0, freq = False))
        bi_smooth_pres.append(train_and_test_model(pos_train_text, neg_train_text, pos_test_text, neg_test_text, n = 2, smoothing_k = 1, freq = False))
        both_smooth_freq.append(train_and_test_unigrams_and_bigrams(pos_train_text, neg_train_text, pos_test_text, neg_test_text, smoothing_k = 1, freq = True))
        both_smooth_pres.append(train_and_test_unigrams_and_bigrams(pos_train_text, neg_train_text, pos_test_text, neg_test_text, smoothing_k = 1, freq = False))        
        both_no_smooth_freq.append(train_and_test_unigrams_and_bigrams(pos_train_text, neg_train_text, pos_test_text, neg_test_text, smoothing_k = 0, freq = True))
        both_no_smooth_pres.append(train_and_test_unigrams_and_bigrams(pos_train_text, neg_train_text, pos_test_text, neg_test_text, smoothing_k = 0, freq = False))




    print("Mean accuracy and variance for:")
    print("Uni, no smooth, freq")
    print(calc_mean_variance(uni_no_smooth_freq))
    print("Uni, smooth, freq")
    print(calc_mean_variance(uni_smooth_freq))
    print("Uni, no smooth, pres")
    print(calc_mean_variance(uni_no_smooth_pres))
    print("Uni, smooth, pres")
    print(calc_mean_variance(uni_smooth_pres))

    print("Bi, no smooth, freq")
    print(calc_mean_variance(bi_no_smooth_freq))
    print("Bi, smooth, freq")
    print(calc_mean_variance(bi_smooth_freq))
    print("Bi, no smooth, pres")
    print(calc_mean_variance(bi_no_smooth_pres))
    print("Bi, smooth, pres")
    print(calc_mean_variance(bi_smooth_pres))

    print("Both, smooth, freq")
    print(calc_mean_variance(both_smooth_freq))
    print("Both, smooth, pres")
    print(calc_mean_variance(both_smooth_pres))
    print("Both, no smooth, freq")
    print(calc_mean_variance(both_no_smooth_freq))
    print("Both, no smooth, pres")
    print(calc_mean_variance(both_no_smooth_pres))


    print("Sign tests")
    print("Smooth vs No Smooth")
    print("Unigrams")
    print("Frequency")
    print("Mean accuracy no smoothing: ", calc_mean_variance(uni_no_smooth_freq))
    print("Mean accuracy smoothing: ", calc_mean_variance(uni_smooth_freq))
    print(sign_test(uni_no_smooth_freq, uni_smooth_freq))
    print("Presence")
    print("Mean accuracy no smoothing: ", calc_mean_variance(uni_no_smooth_pres))
    print("Mean accuracy smoothing: ", calc_mean_variance(uni_smooth_pres))
    print(sign_test(uni_no_smooth_pres, uni_smooth_pres))
    print("Bigrams")
    print("Frequency")
    print("Mean accuracy no smoothing: ", calc_mean_variance(bi_no_smooth_freq))
    print("Mean accuracy smoothing: ", calc_mean_variance(bi_smooth_freq))
    print(sign_test(bi_no_smooth_freq, bi_smooth_freq))
    print("Presence")
    print("Mean accuracy no smoothing: ", calc_mean_variance(bi_no_smooth_pres))
    print("Mean accuracy smoothing: ", calc_mean_variance(bi_smooth_pres))
    print(sign_test(bi_no_smooth_pres, bi_smooth_pres))

    print("Bigrams vs Unigrams")
    print("No Smoothing")
    print("Frequency")
    print("Mean accuracy unigrams: ", calc_mean_variance(uni_no_smooth_freq))
    print("Mean accuracy bigrams: ", calc_mean_variance(bi_no_smooth_freq))
    print(sign_test(uni_no_smooth_freq, bi_no_smooth_freq))
    print("Presence")
    print("Mean accuracy unigrams: ", calc_mean_variance(uni_no_smooth_pres))
    print("Mean accuracy bigrams: ", calc_mean_variance(bi_no_smooth_pres))
    print(sign_test(uni_no_smooth_pres, bi_no_smooth_pres))
    print("Smoothing")
    print("Frequency")
    print("Mean accuracy unigrams: ", calc_mean_variance(uni_smooth_freq))
    print("Mean accuracy bigrams: ", calc_mean_variance(bi_smooth_freq))
    print(sign_test(bi_smooth_freq, uni_smooth_freq))
    print("Presence")
    print("Mean accuracy unigrams: ", calc_mean_variance(uni_smooth_pres))
    print("Mean accuracy bigrams: ", calc_mean_variance(bi_smooth_pres))
    print(sign_test(uni_smooth_pres, bi_smooth_pres))

    print("Freq vs Pres")
    print("Unigrams")
    print("No Smoothing")
    print("Mean accuracy freq: ", calc_mean_variance(uni_no_smooth_freq))
    print("Mean accuracy pres: ", calc_mean_variance(uni_no_smooth_pres))
    print(sign_test(uni_no_smooth_pres, uni_no_smooth_freq))
    print("Smoothing")
    print("Mean accuracy freq: ", calc_mean_variance(uni_smooth_freq))
    print("Mean accuracy pres: ", calc_mean_variance(uni_smooth_pres))
    print(sign_test(uni_smooth_freq, uni_smooth_pres))
    print("Bigrams")
    print("No Smoothing")
    print("Mean accuracy freq: ", calc_mean_variance(bi_no_smooth_freq))
    print("Mean accuracy pres: ", calc_mean_variance(bi_no_smooth_pres))
    print(sign_test(bi_no_smooth_pres, bi_no_smooth_freq))
    print("Smoothing")
    print("Mean accuracy freq: ", calc_mean_variance(bi_smooth_freq))
    print("Mean accuracy pres: ", calc_mean_variance(bi_smooth_pres))
    print(sign_test(bi_smooth_pres, bi_smooth_freq))

    print("Both, with smoothing")
    print("Mean accuracy freq: ", calc_mean_variance(both_smooth_freq))
    print("Mean accuracy pres: ", calc_mean_variance(both_smooth_pres))
    print(sign_test(both_smooth_pres, both_smooth_freq))

    print("Unigrams vs Both")
    print("freq, with smoothing")
    print("Mean accuracy uni: ", calc_mean_variance(uni_smooth_freq))
    print("Mean accuracy both: ", calc_mean_variance(both_smooth_freq))
    print(sign_test(both_smooth_freq, uni_smooth_freq))
    print("freq, no smoothing")
    print("Mean accuracy uni: ", calc_mean_variance(uni_no_smooth_freq))
    print("Mean accuracy both: ", calc_mean_variance(both_no_smooth_freq))
    print(sign_test(both_no_smooth_freq, uni_no_smooth_freq))
    print("Pres, smoothing")
    print("Mean accuracy uni: ", calc_mean_variance(uni_smooth_pres))
    print("Mean accuracy both: ", calc_mean_variance(both_smooth_pres))
    print(sign_test(both_smooth_pres, uni_smooth_pres))
    print("No smoothing, pres")
    print("Mean accuracy uni: ", calc_mean_variance(uni_no_smooth_pres))
    print("Mean accuracy both: ", calc_mean_variance(both_no_smooth_pres))
    print(sign_test(both_no_smooth_pres, uni_no_smooth_pres))


run_tests(10)