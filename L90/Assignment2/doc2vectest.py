from nltk.tokenize import word_tokenize
import os
import random
import math
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


"""
    This code performs some evaluations on the performance of the doc2vec embeddings.
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


def cos_sim(a, b):
    # Calculates cosine similarity between two vectors
    assert len(a) == len(b)
    dot = sum([a[i] * b[i] for i in range(len(a))])
    a_norm = math.sqrt(sum([a_i ** 2 for a_i in a]))
    b_norm = math.sqrt(sum([b_i ** 2 for b_i in b]))
    return dot / (a_norm * b_norm)


def check_doc_sims(d2v_model, pos, neg):
    # Compares similar and non-similar reviews to examine whether the cosine similarity of the embedding vectors
    # increases for similar reviews
    incs = []
    tot = []
    docs = pos + neg
    for i in range(len(docs)):
        for j in range(i, len(docs) - i):
            sim = calc_sim(docs[i], docs[i + j], d2v_model)
            tot.append(sim)
    print("Average similarity between any two reviews:", sum(tot) / len(tot))
    any_two = sum(tot)/len(tot)
    tot = []
    for i in range(0, len(pos) - 2, 3):
        sim = calc_sim(pos[i], pos[i + 1], d2v_model)
        tot.append(sim)
        sim = calc_sim(pos[i], pos[i + 2], d2v_model)
        tot.append(sim)
        sim = calc_sim(pos[i + 1], pos[i + 2], d2v_model)
        tot.append(sim)
    print("Average similarity to pos reviews for same film:", sum(tot)/len(tot))
    inc = (sum(tot)/len(tot) - any_two)/any_two
    print("Increase: ", inc)
    incs.append(inc)
    tot = []
    for i in range(0, len(neg) - 2, 3):
        sim = calc_sim(neg[i], neg[i + 1], d2v_model)
        tot.append(sim)
        sim = calc_sim(neg[i], neg[i + 2], d2v_model)
        tot.append(sim)
        sim = calc_sim(neg[i + 1], neg[i + 2], d2v_model)
        tot.append(sim)
    print("Average similarity to neg reviews for same film:", sum(tot) / len(tot))
    inc = (sum(tot) / len(tot) - any_two) / any_two
    print("Increase: ", inc)
    incs.append(inc)
    tot = []
    for i in range(0, len(pos)):
        for j in range(i, len(pos) - i):
            sim = calc_sim(pos[i], pos[i + j], d2v_model)
            tot.append(sim)
    print("Average similarity between pos reviews:", sum(tot)/len(tot))
    inc = (sum(tot) / len(tot) - any_two) / any_two
    print("Increase: ", inc)
    incs.append(inc)
    tot = []
    for i in range(0, len(neg)):
        for j in range(i, len(neg) - i):
            sim = calc_sim(neg[i], neg[i + j], d2v_model)
            tot.append(sim)
    print("Average similarity between neg reviews:", sum(tot) / len(tot))
    inc = (sum(tot) / len(tot) - any_two) / any_two
    print("Increase: ", inc)
    incs.append(inc)
    tot = []
    for i in range(0, len(pos)):
        k = i // 3
        for j in range(3):
            sim = calc_sim(pos[i], neg[k + j], d2v_model)
            tot.append(sim)
    print("Average similarity between pos and neg reviews for same film:", sum(tot)/len(tot))
    inc = (sum(tot) / len(tot) - any_two) / any_two
    print("Increase: ", inc)
    incs.append(inc)
    print("Av increase: ", sum(incs)/len(incs))


def most_similar_word(doc, d2v_model, k=10):
    # Returns a list of the k words in a review whose vectors are most similar to the review's vector
    sim_list = []
    for word in doc:
        sim = calc_sim(doc, [word], d2v_model)
        sim_list.append([word, sim])
    sim_list.sort(key=lambda x: x[1], reverse=True)
    return sim_list[:k]


def calc_sim(a, b, d2v):
    A = d2v.infer_vector(a)
    B = d2v.infer_vector(b)
    return cos_sim(A, B)


def word_similarity(d2v_model, similar_word_sets):
    # Examines word similarity for words in similarity sets compared to words outside of their similarity sets
    incs = []
    sims = []
    for i in range(len(similar_word_sets)):
        for j in range(len(similar_word_sets[i])):
            for k in range(i, len(similar_word_sets) - i):
                for l in range(len(similar_word_sets[k])):
                    sim = calc_sim([similar_word_sets[i][j]], [similar_word_sets[k][l]], d2v_model)
                    sims.append(sim)
    print("Average similarity outside set: ", sum(sims)/len(sims))
    outside = sum(sims)/len(sims)
    for set in similar_word_sets:
        sims = []
        for i in range(len(set)):
            for j in range(i, len(set) - i):
                sim = calc_sim([set[i]], [set[i + j]], d2v_model)
                sims.append(sim)
        print("Average similarity within set ", set, " : ", sum(sims)/len(sims))
        inc = (sum(sims) / len(sims) - outside) / outside
        print("Increase: ", inc)
        incs.append(inc)
    print("AVerage increase: ", sum(incs)/len(incs))


pos_file_list = [os.path.join("new_test_reviews", "pos", f)
                 for f in os.listdir(os.path.join("new_test_reviews", "pos"))]
neg_file_list = [os.path.join("new_test_reviews", "neg", f)
                 for f in os.listdir(os.path.join("new_test_reviews", "neg"))]
pos, neg = read_files(pos_file_list, neg_file_list)

model_list = {"dbow.model", "dm.model", "dbow100.model", "dm100.model", "dbowlarge.model", "dmlarge.model",
                  "dbow5cutoff.model", "dm5cutoff.model", "dbow1cutoff.model", "dm1cutoff.model"}

similar_word_sets = [["great", "good", "excellent", "amazing"], ["awful", "dreadful", "terrible", "bad"],
                     ["dog", "cat", "puppy", "kitten"], ["director", "actor", "producer", "writer"]]

for model in model_list:
    print(model)
    d2v_model = Doc2Vec.load(model)
    check_doc_sims(d2v_model, pos, neg)
    for doc in pos + neg:
        print(most_similar_word(doc, d2v_model))
        print("----------")
    word_similarity(d2v_model, similar_word_sets)
