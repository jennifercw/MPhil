from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import os
import random


"""
    This code trains a variety of doc2vec embeddings for use with movie review classification.
    The doc2vec models produced by this code and used to obtain the results found in my writeup, along with the 
    saved best models for each setup can be downloaded here: 
    https://drive.google.com/file/d/1wCk_b-efz4r9xBGX7FD4NqT0x9-usYXi/view?usp=sharing
"""


def read_data():
    # Reads in the training data and strips it of <br/> tags
    data = []
    file_list = [os.path.join("aclImdb_v1", "aclImdb", "train", "neg", f) for f in
                 os.listdir(os.path.join("aclImdb_v1", "aclImdb", "train", "neg"))]
    file_list.extend([os.path.join("aclImdb_v1", "aclImdb", "train", "pos", f) for f in
                 os.listdir(os.path.join("aclImdb_v1", "aclImdb", "train", "pos"))])
    file_list.extend([os.path.join("aclImdb_v1", "aclImdb", "test", "pos", f) for f in
                      os.listdir(os.path.join("aclImdb_v1", "aclImdb", "test", "pos"))])
    file_list.extend([os.path.join("aclImdb_v1", "aclImdb", "test", "neg", f) for f in
                      os.listdir(os.path.join("aclImdb_v1", "aclImdb", "test", "neg"))])

    for fname in file_list:
        with open(fname, 'r') as f:
            text = f.read().lower()
            text = text.replace('<br />', ' ')
            data.append(text)
    random.shuffle(data)
    return data


def tag_data(data):
    tagged_data = [TaggedDocument(words=word_tokenize(d.lower()), tags=[str(i)]) for i, d in enumerate(data)]
    return tagged_data


def train_model(documents, epochs, embedding_dim, learning_rate, min_learning_rate, freq_cutoff, neg, dm, model_name):
    # Trains model with given parameters
    model = Doc2Vec(vector_size=embedding_dim,
                    alpha=learning_rate,
                    min_alpha=min_learning_rate,
                    min_count=freq_cutoff,
                    negative=neg,
                    dm=dm,
                    workers=4)
    model.build_vocab(documents)
    for ep in range(epochs):
        print("Epoch ", ep)
        random.shuffle(documents)
        model.train(documents=documents, total_examples=model.corpus_count, epochs=model.iter)
    model.save(model_name)
    return model


reviews = read_data()
tagged_reviews = tag_data(reviews)
mod = train_model(tagged_reviews, epochs=50, embedding_dim=100, learning_rate=0.005, min_learning_rate=0.00025,
                  freq_cutoff=3, neg=5, dm=0, model_name="dbow.model")
mod = train_model(tagged_reviews, epochs=50, embedding_dim=100, learning_rate=0.005, min_learning_rate=0.00025,
                  freq_cutoff=3, neg=5, dm=1, model_name="dm.model")
mod = train_model(tagged_reviews, epochs=100, embedding_dim=100, learning_rate=0.005, min_learning_rate=0.00025,
                  freq_cutoff=3, neg=5, dm=0, model_name="dbow100.model")
mod = train_model(tagged_reviews, epochs=100, embedding_dim=100, learning_rate=0.005, min_learning_rate=0.00025,
                  freq_cutoff=3, neg=5, dm=0, model_name="dm100.model")
mod = train_model(tagged_reviews, epochs=50, embedding_dim=200, learning_rate=0.005, min_learning_rate=0.00025,
                  freq_cutoff=3, neg=5, dm=0, model_name="dbowlarge.model")
mod = train_model(tagged_reviews, epochs=50, embedding_dim=200, learning_rate=0.005, min_learning_rate=0.00025,
                  freq_cutoff=3, neg=5, dm=1, model_name="dmlarge.model")
mod = train_model(tagged_reviews, epochs=50, embedding_dim=100, learning_rate=0.005, min_learning_rate=0.00025,
                  freq_cutoff=5, neg=5, dm=0, model_name="dbow5cutoff.model")
mod = train_model(tagged_reviews, epochs=50, embedding_dim=100, learning_rate=0.005, min_learning_rate=0.00025,
                  freq_cutoff=5, neg=5, dm=1, model_name="dm5cutoff.model")
mod = train_model(tagged_reviews, epochs=50, embedding_dim=100, learning_rate=0.005, min_learning_rate=0.00025,
                  freq_cutoff=1, neg=5, dm=0, model_name="dbow1cutoff.model")
mod = train_model(tagged_reviews, epochs=50, embedding_dim=100, learning_rate=0.005, min_learning_rate=0.00025,
                  freq_cutoff=1, neg=5, dm=1, model_name="dm1cutoff.model")
