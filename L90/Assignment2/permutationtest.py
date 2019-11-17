import random
import pickle


def permutation_test(a, b):
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    diff = abs(mean_b - mean_a)
    assert(len(a) == len(b))
    n = len(a)
    s = 0
    R = 5000
    for r in range(R):
        perm_a = []
        perm_b = []
        for i in range(n):
            flip = random.randint(0,1)
            if flip:
                perm_b.append(a[i])
                perm_a.append(b[i])
            else:
                perm_a.append(a[i])
                perm_b.append(b[i])
        new_mean_a = sum(perm_a)/len(perm_a)
        new_mean_b = sum(perm_b)/len(perm_b)
        new_diff = abs(new_mean_b - new_mean_a)
        if new_diff > diff:
            s += 1
    return (s + 1) / (R + 1)


def compare(a, b, a_name, b_name):
    print("Average accuracy, ", a_name, sum(a)/len(a))
    print("Average accuracy, ", b_name, sum(b)/len(b))
    print("p = ", permutation_test(a, b))


def run_comparisons():
    # For d2v: Within each kernel which embedding gives best?
    # Then compare best one from each kernel?
    # For others: which kernel is best for each of unifreq, unipres, etc
    # Then uni v bi, pres v freq
    d2v_accuracy = pickle.load(open("d2vaccuracy.p", "rb"))
    svm_accuracy = pickle.load(open("svm_accuracy.p", "rb"))
    for name, accs in svm_accuracy.items():
        print(name)
        print(sum(accs)/len(accs))
    for kernel, results in d2v_accuracy.items():
        for name, accs in results.items():
            print(kernel, name)
            print(sum(accs)/len(accs))

test = pickle.load(open("test_accs.p", "rb"))
test_d2v = pickle.load(open("test_accs_d2v.p", "rb"))
print(test)
print(test_d2v)
run_comparisons()