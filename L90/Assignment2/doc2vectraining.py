from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import os
import random

"""
The models that I obtained from this code and that I used to obtain my results can be downloaded from:
    https://drive.google.com/file/d/1ykdC4qwMj-nZC6t5P5KJQZespIaH__7-/view?usp=sharing
"""

def read_data():
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
    # Strip HTML tags as well?
    tagged_data = [TaggedDocument(words=word_tokenize(d.lower()), tags=[str(i)]) for i, d in enumerate(data)]
    return tagged_data

def train_model(documents, epochs, embedding_dim, learning_rate, min_learning_rate, freq_cutoff, neg, dm, model_name):
    model = Doc2Vec(vector_size=embedding_dim,
                    alpha=learning_rate,
                    min_alpha=min_learning_rate,
                    min_count=freq_cutoff,
                    negative=neg,
                    dm=dm,
                    workers=4)
    model.build_vocab(documents)
    for ep in range(epochs):
        print(ep)
        random.shuffle(documents)
        model.train(documents=documents, total_examples=model.corpus_count, epochs=model.iter)
    model.save(model_name)
    return model

reviews = read_data()
print(reviews[:5])
tagged_reviews = tag_data((reviews))
print(tagged_reviews[0])
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