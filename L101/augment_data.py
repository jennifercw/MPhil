import pandas as pd
import random
import pickle

def load_data(data_path='trainset.csv'):
    pdread = pd.read_csv(data_path)
    mr = pdread['mr'].tolist()
    sentences = pdread['ref'].tolist()
    mr_dict = {}
    for m in mr:
        slots = m.split(", ")
        for slot in slots:
            slot = slot[:-1]
            slot_data = slot.split('[')
            if slot_data[0] not in mr_dict:
                mr_dict[slot_data[0]] = set()
            mr_dict[slot_data[0]].add(slot_data[1])
    return mr, sentences, mr_dict


def analyse_data(mrs, sentence_list, mr_dict):
    assert(len(mrs) == len(sentence_list))
    analysed_data = {}
    for i in range(len(mrs)):
        analysed_data[(mrs[i], sentence_list[i])] = {}
        for slot in mr_dict.keys():
            if mrs[i].find(slot) == -1:
                analysed_data[(mrs[i], sentence_list[i])][slot] = False
            else:
                analysed_data[(mrs[i], sentence_list[i])][slot] = True
    return analysed_data


def augment_data(mr_dict, analysed):
    augmented = {}
    for data, slots in analysed.items():
        missing = []
        for slot in slots.keys():
            if slots[slot] is False:
                missing.append(slot)
        if len(missing) == 0:
            continue
        slot_to_add = random.choice(missing)
        slot_value = random.sample(mr_dict[slot_to_add], 1)[0]
        new_mr = data[0] + ", " + slot_to_add + "[" + slot_value + "]"
        new_slots = slots
        new_slots[slot_to_add] = True
        augmented[(new_mr, data[1])] = new_slots
    return augmented


def remove_data(analysed):
    removed = {}
    for data, slots in analysed.items():
        pres_slots = [slot for slot in slots.keys() if slot != "name"]
        to_remove_slot = random.choice(pres_slots)
        new_mr = remove_slot(data[0], to_remove_slot)
        new_slots = slots
        new_slots[to_remove_slot] = False
        removed[(new_mr, data[1])] = new_slots
    return removed


def remove_slot(mr, to_remove):
    start = mr.find(to_remove)
    next_comma = mr[start:].find(",")
    if next_comma == -1:
        end = len(mr)
        start -= 2
    else:
        end = start + next_comma + 2
    mr = mr[:start] + mr[end:]
    return mr


mrs, sentence_list, mr_dict = load_data()
print(mr_dict)
analysed = analyse_data(mrs, sentence_list, mr_dict)
print(len(analysed))
removed = remove_data(analysed)
augmented = augment_data(mr_dict, analysed)
aug_data_list = [(data, 1) for data in analysed.keys()] + [(data, 0) for data in augmented.keys()]
rem_data_list = [(data, 1) for data in analysed.keys()] + [(data, 0) for data in removed.keys()]
random.shuffle(aug_data_list)
random.shuffle(rem_data_list)
pickle.dump(aug_data_list, open('aug_data.obj', 'wb'))
pickle.dump(aug_data_list, open('rem_data.obj', 'wb'))
# Do I want to train individual classifiers for each slot?
