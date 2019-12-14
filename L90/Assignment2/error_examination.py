import pickle


"""
    This code can be used to see all of the misclassifications from a given model on the blind test set so that they can
    be examined and classified.
"""

incorr = pickle.load(open("incorr.p", "rb"))
incorr_d2v = pickle.load(open("incorr_d2v.p", "rb"))
neg_pos = ["NEGATIVE", "POSITIVE"]

model = "dbow5cutoff.model"
kernel = "sigmoid"
errors = incorr_d2v[kernel][model]

print(len(errors))
for error in errors:
    print("----------------")
    print("Correct classification: ", neg_pos[error[1]])
    print("Review:")
    print(' '.join(error[0]))