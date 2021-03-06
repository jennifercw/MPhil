import random
import pickle


"""
    This code uses the permutation test to perform statistical significance testing on the results of the 
    cross-validation tests.
"""


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


def calc_mean_variance(acc):
    # Given a list of accuracy values, calculates their mean and variance.
    m = sum(acc) / len(acc)
    v = sum((x - m) ** 2 for x in acc) / len(acc)
    return m, v


def run_comparisons():
    d2v_accuracy = pickle.load(open("d2vaccuracy.p", "rb"))
    svm_accuracy = pickle.load(open("svm_accuracy.p", "rb"))

    for kern in d2v_accuracy.keys():
        for model in d2v_accuracy[kern].keys():
            accs = d2v_accuracy[kern][model]
            print(kern, model)
            m, v = calc_mean_variance(accs)
            print(m*100, v*100)

    for model in svm_accuracy.keys():
        accs = svm_accuracy[model]
        print(model)
        m, v = calc_mean_variance(accs)
        print(m * 100, v * 100)
        
    print("Non doc2vec models")
    print("unifreqrbf vs unifreqlin")
    compare(svm_accuracy["unifreq"], svm_accuracy["unifreqlin"], "unifreqrbf", "unifreqlin")

    print("unipresrbf vs unipreslin")
    compare(svm_accuracy["unipres"], svm_accuracy["unipreslin"], "unipresrbf", "unipreslin")

    print("bifreqrbf vs bifreqlin")
    compare(svm_accuracy["bifreq"], svm_accuracy["bifreqlin"], "bifreqrbf", "bifreqlin")

    print("bipresrbf vs bipressig")
    compare(svm_accuracy["bipres"], svm_accuracy["bipressig"], "bipresrbf", "bipressig")

    print("unipressig vs unifreqlin")
    compare(svm_accuracy["unipressig"], svm_accuracy["unifreqlin"], "unipressig", "unifreqlin")

    print("Doc2vec models")

    print("dbowsig vs dbowlin")
    compare(d2v_accuracy["rbf"]["dbow.model"], d2v_accuracy["sigmoid"]["dbow.model"], "dbowrbf", "dbowsig")
    print("dbowrbf vs dbowpoly")
    compare(d2v_accuracy["rbf"]["dbow.model"], d2v_accuracy["poly"]["dbow.model"], "dbowrbf", "dbowpoly")

    print("dmrbf vs dmlin")
    compare(d2v_accuracy["rbf"]["dm.model"], d2v_accuracy["linear"]["dm.model"], "dmrbf", "dmlin")
    print("dmpoly vs dmlin")
    compare(d2v_accuracy["poly"]["dm.model"], d2v_accuracy["linear"]["dm.model"], "dmpoly", "dmlin")

    print("dbowlargepoly vs dbowlargesig")
    compare(d2v_accuracy["poly"]["dbowlarge.model"], d2v_accuracy["sigmoid"]["dbowlarge.model"], "dbowlargepoly",
            "dbowlargesig")

    print("dbow100sig vs dbow100rbf")
    compare(d2v_accuracy["sigmoid"]["dbow100.model"], d2v_accuracy["rbf"]["dbow100.model"], "dbow100sig",
            "dbow100rbf")

    print("dm100rbf vs dm100lin")
    compare(d2v_accuracy["rbf"]["dm100.model"], d2v_accuracy["linear"]["dm100.model"], "dm100rbf",
            "dm100lin")

    print("dmlargesig vs dmlargelin")
    compare(d2v_accuracy["sigmoid"]["dmlarge.model"], d2v_accuracy["linear"]["dmlarge.model"], "dmlargesig",
            "dmlargelin")

    print("dbow5cutoffpoly vs dbow5cutoffsig")
    compare(d2v_accuracy["poly"]["dbow5cutoff.model"], d2v_accuracy["sigmoid"]["dbow5cutoff.model"], "dbow5cutoffpoly",
            "dbow5cutoffsig")

    print("dm5cutoffrbf vs dbow5cutoffsig")
    compare(d2v_accuracy["rbf"]["dm5cutoff.model"], d2v_accuracy["sigmoid"]["dm5cutoff.model"], "dm5cutoffrbf",
            "dm5cutoffsig")

    print("dbow1cutoffsig vs dbow1cutoffpoly")
    compare(d2v_accuracy["sigmoid"]["dbow1cutoff.model"], d2v_accuracy["poly"]["dbow1cutoff.model"], "dbow1cutoffsig",
            "dbow1cutoffpoly")

    print("dm1cutoffrbf vs dm1cutofflin")
    compare(d2v_accuracy["rbf"]["dm1cutoff.model"], d2v_accuracy["linear"]["dm1cutoff.model"], "dm1cutoffrbf",
            "dm1cutofflin")

    print("dbowlargepoly vs dbow5cutoffrbf")
    compare(d2v_accuracy["poly"]["dbowlarge.model"], d2v_accuracy["rbf"]["dbow5cutoff.model"], "dbowlargepoly",
            "dbow5cutoffrbf")

    print("dbowlargepoly vs dbowrbf")
    compare(d2v_accuracy["poly"]["dbowlarge.model"], d2v_accuracy["rbf"]["dbow.model"], "dbowlargepoly",
            "dbowrbf")


test = pickle.load(open("test_accs.p", "rb"))
test_d2v = pickle.load(open("test_accs_d2v.p", "rb"))
print(test)
print(test_d2v)
run_comparisons()