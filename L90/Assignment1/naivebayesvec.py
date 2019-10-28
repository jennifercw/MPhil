from collections import Counter
import os
import math

# TODO: SOMETHING IS BROKEN FOR NO SMOOTHING.

def read_round_robin_groups(k = 10):
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
    all_files = pos_files + neg_files
    all_vocab = Counter()
    for f in all_files:
        for i in range(len(f) - n + 1):
            w = " ".join([f[i + j] for j in range(n)])
            all_vocab[w] += 1
    cutoff = sum([1 for i in all_vocab.items() if i[1] >= 5])
    top_vocab = [v[0] for v in all_vocab.most_common(cutoff)]
    top_vocab.append(" ".join(["UNK"] * n))
    #vocab_set = set((top_vocab[i] for i in range(len(top_vocab))))
    vocab_dict = {top_vocab[i] : i for i in range(len(top_vocab))}
    reverse_dict = {c : v for v, c in vocab_dict.items()}
    return vocab_dict, reverse_dict

def get_stats(pos_files, neg_files, vocab_dict, n = 1, smoothing_k = 1):
    pos_neg_likelihood = {}
    total_pos = len(pos_files)
    total_neg = len(neg_files)
    pos_neg_likelihood["pos"] = total_pos / (total_pos + total_neg)
    pos_neg_likelihood["neg"] = total_neg / (total_pos + total_neg)

    pos_count = sum([len(rev) - n + 1 for rev in pos_files])
    neg_count = sum([len(rev) - n + 1 for rev in neg_files])
    word_counts = {"pos" : {}, "neg" : {}}
    word_probs = {"pos" : {}, "neg" : {}}

    for w in vocab_dict.keys():
        word_counts["pos"][w] = smoothing_k
        word_counts["neg"][w] = smoothing_k
    for p in pos_files:
        for i in range(len(p) - n + 1):
            w = " ".join([p[i + j] for j in range(n)])
            if w not in vocab_dict:
                w = " ".join(["UNK"] * n)
            word_counts["pos"][w] += 1
    for w, count in word_counts["pos"].items():
        word_probs["pos"][w] = count / (pos_count + (smoothing_k * len(vocab_dict)))

    for ng in neg_files:
        for i in range(len(ng) - n + 1):
            w = " ".join([ng[i + j] for j in range(n)])
            if w not in vocab_dict:
                w = " ".join(["UNK"] * n)
            word_counts["neg"][w] += 1
    for w, count in word_counts["neg"].items():
        word_probs["neg"][w] = count / (neg_count + (smoothing_k * len(vocab_dict)))

    return pos_neg_likelihood, word_probs


def vectorise(rev, vocab_dict, n = 1, freq = True):
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
    return fact(n)/(fact(k) * fact(n - k))


def fact(x):
    if x == 1 or x == 0:
        return 1
    else:
        return x * fact(x - 1)


def sign_test(acc1, acc2, q = 0.5):
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
    sign_test = 2 * sum([binom_coeff(N, i) * q**i * (1 - q)**(N-i)  for i in range(k + 1)])
    print(sign_test)

def train_and_test_model(pos, neg, pos_test, neg_test, n = 1, smoothing_k = 1, freq = True):
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
    print(acc)
    return(acc)

def calc_mean_variance(acc):
    m = sum(acc)/len(acc)
    v = sum((x-m)**2 for x in acc)/len(acc)
    return m, v

def run_tests(k = 10):
    pos_robins, neg_robins = read_round_robin_groups()
    uni_no_smooth_freq = []
    uni_smooth_freq = []
    uni_no_smooth_pres = []
    uni_smooth_pres = []
    bi_no_smooth_freq = []
    bi_smooth_freq = []
    bi_no_smooth_pres = []
    bi_smooth_pres = []

    for i in range(k):
        # Divide train and test sets
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


run_tests(10)