import pickle
import math

data = pickle.load(open('data.obj', 'rb'))
print(data)
print(len(data))
train_test_split = 0.8
cutoff = math.floor(train_test_split * len(data))
train_data = data[:cutoff]
test_data = data[cutoff:]