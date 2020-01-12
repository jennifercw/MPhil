
def calc_micro(filename=""):
	summ = 0
	prec = 0
	rec = 0
	f1 = 0
	f = open(filename, 'r')
	for line in f.readlines():
		split = line.split(' ')
		split = [s for s in split if s != '']
		if len(split) < 4:
			continue
		if split[0] != '-':
			precision = float(split[0])
		else:
			precision = 0
		if split[1] != '-':
			recall = float(split[1])
		else:
			recall = 0
		if split[2] != '-':
			f1_idv = float(split[2])
		else:
			f1_idv = 0
		num = float(split[3])
		summ += num
		prec += (num * precision)
		rec += (num * recall)
		f1 += (num * f1_idv)
	print("Precision: " , prec/summ)
	print("Recall: ", rec/summ)
	print("f1: ", f1/summ)


def calc_macro(filename=""):
	summ = 0
	prec = 0
	rec = 0
	f1 = 0
	f = open(filename, 'r')
	for line in f.readlines():
		split = line.split(' ')
		split = [s for s in split if s != '']
		if len(split) < 4:
			continue
		if split[0] != '-':
			precision = float(split[0])
		else:
			precision = 0
		if split[1] != '-':
			recall = float(split[1])
		else:
			recall = 0
		if split[2] != '-':
			f1_idv = float(split[2])
		else:
			f1_idv = 0
		summ += 1
		prec += precision
		rec += recall
		f1 += f1_idv
	print("Precision: " , prec/summ)
	print("Recall: ", rec/summ)
	print("f1: ", f1/summ)

print("SR Micro")
calc_micro("sr_results.txt")
print("SR Macro")
calc_macro("sr_results.txt")
print("NN Micro")
calc_micro("nn_results.txt")
print("NN Macro")
calc_macro("nn_results.txt")
