import json
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "..")  # make sure to import utils and config successfully
from utils.data import *
from utils.glove import *
from utils import path
# from utils import store_result
from GAS.config import MemNNConfig
from GAS.model_4 import Modelmy
from GAS.loss_func import Marginal_loss
from GAS.test import *
_path = path.WSD_path()
config = MemNNConfig()


print('Loading all-words task data...')
train_dataset = _path.ALL_WORDS_TRAIN_DATASET[0]  # semcor
print('train_dataset: ' + train_dataset)
val_dataset = _path.ALL_WORDS_VAL_DATASET  # semeval2007
print('val_dataset: ' + val_dataset)
test_dataset = _path.ALL_WORDS_TEST_DATASET[1]  # senseval2
print('test_dataset: ' + test_dataset)

train_data = load_train_data(train_dataset)
val_data = load_val_data(val_dataset)
test_data = load_test_data(test_dataset)
o_len = len(test_data)
print('O Original dataset size (train/val/test): %d / %d / %d' % (len(train_data), len(val_data), len(test_data)))

# === Using back-off strategy
back_off_result = []
# 确定有train_dataset数据集
if train_dataset in _path.ALL_WORDS_TRAIN_DATASET:
    # 记录testdata的长度
    test_data_lenth_pre = len(test_data)

    train_data, test_data, word_to_senses, back_off_result, backoff_len, backoff_right = data_postprocessing(
        train_dataset, test_dataset, train_data, test_data, 'FS', config.min_sense_freq, config.max_n_sense)
    val_data = data_postprocessing_for_validation(val_data, word_to_senses)
    # 筛选后dataset
    print('1 Filtered dataset size (train/val/test): %d / %d / %d' % (
        len(train_data), len(val_data), len(test_data)))
    # back_off_result 存储与new_test_data中target word重合的部分
    print('***Test using back-off instance: %d' % (len(back_off_result)))
    # 丢弃的test 数量
    missed = test_data_lenth_pre - (len(test_data) + len(back_off_result))
    # 丢失率
    missed_ratio = float(missed) / test_data_lenth_pre
    #
    print('***Test missed instance(not in MFS/FS): %d/%d = %.3f' % (
        (missed, test_data_lenth_pre, missed_ratio)))
# print(back_off_result)

# === Build vocab utils
word_to_id = build_vocab(train_data)
# 字典的大小
config.vocab_size = len(word_to_id)
print('Vocabulary size: %d' % len(word_to_id))

target_word_to_id, target_sense_to_id, n_senses_from_target_id, word_to_sense = build_sense_ids(word_to_senses)
old_word_to_sense = word_to_sense
# senses的数量
tot_n_senses = sum(n_senses_from_target_id.values())
# 平均每个单词几个sense
average_sense = float(tot_n_senses) / len(n_senses_from_target_id)
assert average_sense >= 2.0  # ambiguous word must have two sense
print('Avg n senses per target word: %.4f' % average_sense)
with open('../tmp/target_word.txt', 'w') as f:
    for word, id in target_word_to_id.items():
        f.write('{}\t{}\n'.format(word, id))

train_data = convert_to_numeric(train_data, word_to_id, target_word_to_id, target_sense_to_id, mode='Train')
val_data = convert_to_numeric(val_data, word_to_id, target_word_to_id, target_sense_to_id, mode='Val')
test_data = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, mode='Test')
print('2 After convert_to_numeric dataset size (train/val/test): %d / %d / %d' % (
    len(train_data), len(val_data), len(test_data)))
# {id : word}
target_id_to_word = {id: word for (word, id) in target_word_to_id.items()}
# [{sense_id : sense...},...]
# target_id_to_sense_id_to_sense = [{sense_id: sense for (sense, sense_id) in sense_to_id.items()} for
#                                   (target_id, sense_to_id) in enumerate(target_sense_to_id)]

# get dic and make numeric
gloss_dict = load_dictionary(train_dataset, target_word_to_id, config.gloss_expand_type)
"""
    {'newspaper%1:06:00::': 'the physical object that is the product of a newspaper publisher'}
"""
mask_size = 2 * config.n_lstm_units

dict_data, config.max_n_sense = bulid_dictionary_id(gloss_dict, target_sense_to_id, word_to_id, word_to_id['<pad>'],
                                                    mask_size, config.max_gloss_words)

auxiliary_metrix, gloss_dict = create_auxiliary_metrix()
gloss_id = gloss_to_id(gloss_dict, word_to_id)  # (2700,100)