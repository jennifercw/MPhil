import pandas as pd
import pickle
from collections import Counter
from nltk import word_tokenize

start_char = '\t'
end_char = '\n'
train_set = pd.read_csv("trainset.csv")
test_set = pd.read_csv("testset.csv")
dev_set = pd.read_csv("devset.csv")
inputs = train_set['mr'].tolist() + test_set['MR'].tolist() + dev_set['mr'].tolist()
outputs = train_set['ref'].tolist() + dev_set['ref'].tolist()
characters = set()
for i in range(len(inputs)):
    for char in inputs[i]:
        if char not in characters:
            characters.add(char)
for i in range(len(outputs)):
    outputs[i] = start_char + outputs[i] + end_char
    for char in outputs[i]:
        if char not in characters:
            characters.add(char)
need_included = [str(i) for i in range(10)] + [chr(i) for i in range(65, 123)] + \
                "!\"£$%^&*()_+-={}[]:@~;'#<>?,./".split()

for c in need_included:
    if c not in characters:
        characters.add(c)

pickle.dump(characters, open("char_set.obj", "wb"))

mr_words = Counter()
sentence_words = Counter()
for i in range(len(inputs)):
    for w in word_tokenize(inputs[i]):
        mr_words[w] += 1
for i in range(len(outputs)):
    for w in word_tokenize(outputs[i]):
        sentence_words[w] += 1

print(sentence_words.most_common(10))
sentence_word_set = set(sentence_words)
sentence_word_set.add('\t')
sentence_word_set.add('\n')

pickle.dump(set(mr_words), open("mr_word_set.obj", "wb"))
pickle.dump(sentence_word_set, open("ref_word_set.obj", "wb"))
