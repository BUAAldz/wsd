import json
import torch
path = "./include_target.json"
file = open(path, "r")
sense_dict = {}
gloss_dict = {}
word_dict = {}
include_dict = {}
sense_to_gloss = {}
word_to_sense = {}
for line in file:
    temp_dict = json.loads(line)
    if temp_dict['target_word'] not in word_dict:
        word_dict[temp_dict['target_word']] = len(word_dict)
    if temp_dict['target_word'] not in word_to_sense:
        word_to_sense[temp_dict['target_word']] = []
    elif temp_dict['sense'] not in word_to_sense[temp_dict['target_word']]:
        word_to_sense[temp_dict['target_word']].append(temp_dict['sense'])
    if temp_dict['sense'] not in sense_dict:
        sense_dict[temp_dict['sense']] = len(sense_dict)
        sense_to_gloss[temp_dict['sense']] = temp_dict['gloss']
    if temp_dict['gloss'] not in gloss_dict:
        gloss_dict[temp_dict['gloss']] = len(gloss_dict)
        include_dict[temp_dict['gloss']] = temp_dict['include_target']

sense_to_gloss_id = torch.zeros(len(sense_dict)) #(2979)
word_to_sense_id = torch.zeros((len(word_dict), 10)) #(653,10)
gloss_to_word_id = torch.zeros((len(gloss_dict), 6)) #(2700,6)
gloss_to_word_index = torch.zeros((len(gloss_dict), 6)) #(2700,6)

n = 0
for i in sense_dict:
    sense_to_gloss_id[n] = gloss_dict[sense_to_gloss[i]]
    n += 1


n = 0
for i in word_dict:
    m = 0
    for j in word_to_sense[i]:
        word_to_sense_id[n] = sense_dict[j]
        m += 1
    n += 1

n = 0
for i in gloss_dict:
    m = 0
    index = 0
    for w in i.split(' '):
        if w in word_dict:
            gloss_to_word_id[n][m] = word_dict[w]
            gloss_to_word_index[n][m] = index
            m += 0
        index += 1
    n += 1

