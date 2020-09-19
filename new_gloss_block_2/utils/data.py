# -*- coding: utf-8 -*-

import lxml.etree as et
import math
import numpy as np
import collections
import re
import random
from bs4 import BeautifulSoup
from bs4 import NavigableString
import pickle
from utils import path
# import path
import config
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from nltk.stem import WordNetLemmatizer
import torch
import json
wordnet_lemmatizer = WordNetLemmatizer()  # download wordnet: import nltk; nltk.download("wordnet") in readme.txt

_path = path.WSD_path()
config = config.MemNNConfig()
wn = WordNetCorpusReader(_path.WORDNET_PATH, '.*')
print('data.py:','wordnet version %s: %s' % (wn.get_version(), _path.WORDNET_PATH))

path_words_notin_vocab = '../tmp/words_notin_vocab_{}.txt'

pos_dic = {
    'ADJ': u'a',
    'ADV': u'r',
    'NOUN': u'n',
    'VERB': u'v', }

POS_LIST = pos_dic.values()  # ['a', 'r', 'n', 'v']


def load_train_data(dataset):
    if dataset in _path.LS_DATASET: # no use
        return load_lexical_sample_data(_path.LS_TRAIN_PATH.format(dataset), True)
    elif dataset in _path.ALL_WORDS_TRAIN_DATASET: # ['semcor', 'semcor+omsti']
        # '../data/All_Words_WSD/Training_Corpora/{0}/{0}.data.xml'
        # '../data/All_Words_WSD/Training_Corpora/{0}/{0}.gold.key.txt'
        # '../data/All_Words_WSD/Training_Corpora/{0}/{0}.dict.xml
        return load_all_words_data(_path.ALL_WORDS_TRAIN_PATH.format(dataset),
                                   _path.ALL_WORDS_TRAIN_KEY_PATH.format(dataset),
                                   _path.ALL_WORDS_DIC_PATH.format(dataset), True)
    else:
        raise ValueError('%s, %s. Provided: %s' % (
            ','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TRAIN_DATASET), dataset))


def load_val_data(dataset):
    if dataset in _path.LS_DATASET: # no use
        return load_lexical_sample_data(_path.LS_VAL_PATH.format(dataset), True)
    elif dataset in _path.ALL_WORDS_TEST_DATASET:

        # '../data/All_Words_WSD/Evaluation_Datasets/{0}/{0}.data.xml'
        # '../data/All_Words_WSD/Evaluation_Datasets/{0}/{0}.gold.key.txt'
        # '../data/All_Words_WSD/Evaluation_Datasets/{0}/{0}.gold.key.withPos.txt'
        #
        return load_all_words_data(_path.ALL_WORDS_TEST_PATH.format(dataset),
                                   _path.ALL_WORDS_TEST_KEY_PATH.format(dataset), None, False)
    else:
        raise ValueError('%s, %s. Provided: %s' % (
            ','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TEST_DATASET), dataset))


def load_test_data(dataset):
    if dataset in _path.LS_DATASET:
        return load_lexical_sample_data(_path.LS_TEST_PATH.format(dataset), False)
    elif dataset in _path.ALL_WORDS_TEST_DATASET:
        return load_all_words_data(_path.ALL_WORDS_TEST_PATH.format(dataset),
                                   _path.ALL_WORDS_TEST_KEY_PATH.format(dataset), None, False)
    else:
        raise ValueError('%s, %s. Provided: %s' % (
            ','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TEST_DATASET), dataset))


def load_lexical_sample_data(path, is_training=None):
    data = []
    doc = BeautifulSoup(open(path), 'xml')
    instances = doc.find_all('instance')

    for instance in instances:
        answer = None
        context = None
        for child in instance.children:
            if isinstance(child, NavigableString):
                continue
            if child.name == 'answer':
                senseid = child.get('senseid')
                if senseid == 'P' or senseid == 'U':
                    pass
                elif not answer:
                    answer = senseid
            elif child.name == 'context':
                context = child.prettify()
            else:
                print(child.name)
                print(instance.text)
                raise ValueError('unknown child tag to instance')

        def clean_context(ctx_in, has_target=False):
            replace_target = re.compile("<head.*?>.*</head>")
            replace_newline = re.compile("\n")
            replace_dot = re.compile("\.")
            replace_cite = re.compile("'")
            replace_frac = re.compile("[\d]*frac[\d]+")
            replace_num = re.compile("\s\d+\s")
            rm_context_tag = re.compile('<.{0,1}context>')
            rm_cit_tag = re.compile('\[[eb]quo\]')
            rm_misc = re.compile("[\[\]\$`()%/,\.:;-]")

            ctx = replace_newline.sub(' ', ctx_in)  # (' <eop> ', ctx)
            if not has_target:
                ctx = replace_target.sub(' <target> ', ctx)

            ctx = replace_dot.sub(' ', ctx)  # .sub(' <eos> ', ctx)
            ctx = replace_cite.sub(' ', ctx)  # .sub(' <cite> ', ctx)
            ctx = replace_frac.sub(' <frac> ', ctx)
            ctx = replace_num.sub(' <number> ', ctx)
            ctx = rm_cit_tag.sub(' ', ctx)
            ctx = rm_context_tag.sub('', ctx)
            ctx = rm_misc.sub('', ctx)

            word_list = [word for word in re.split('`|, | +|\? |! |: |; |\(|\)|_|,|\.|"|“|”|\'|\'', ctx.lower()) if word]
            return word_list

        # if valid
        if (is_training and answer and context) or (not is_training and context):
            context = clean_context(context)
            lemma = instance.get('id').split('.')[0]
            pos = instance.get('id').split('.')[1]
            if pos in POS_LIST:
                word = lemma + '#' + pos
            else:
                word = lemma
            pos_list = ['<pad>'] * len(context)
            x = {
                'id': instance.get('id'),
                'context': context,
                'target_sense': answer,  # don't support multiple answers
                'target_word': word,
                'poss': pos_list,
            }

            data.append(x)

    return data


def load_all_words_data(data_path, key_path=None, dic_path=None, is_training=False):
    """
    按照一定的格式读取数据集
    :param data_path: ...data.xml
    :param key_path: ...key.txt
    :param dic_path: ..dict.xml
    :param is_training:
    :return:  data list
    """

    """
    dict.xml 数据格式
        <lexelt item="shotgun#n" pos="n" sence_count_wn="1" sense_count_corpus="1" word_example_count="2">
         <sense gloss="firearm that is a double-barreled smoothbore shoulder weapon for firing shot at short ranges" id="shotgun%1:06:00::" sense_example_count="2" sense_freq="2" synset="shotgun scattergun">
         </sense>
        </lexelt>
    """
    # 存储target 的词义信息
    word_count_info = {}

    if dic_path:  # train_data 有效
        # 利用BeautifulSoup 解析dict.xml
        soup = BeautifulSoup(open(dic_path), 'lxml')
        # 遍历所有的lexelt标签
        for lexelt_tag in soup.find_all('lexelt'):
            lemma = lexelt_tag['item'] # word+#+pos
            sense_count_wn = int(lexelt_tag['sence_count_wn']) #
            sense_count_corpus = int(lexelt_tag['sense_count_corpus'])   # 词义数量
            word_count_info[lemma] = [sense_count_wn, sense_count_corpus] # key lemma ->[count1,count2]

    # data.xml instance id -> dict.xml sens id
    id_to_sensekey = {}
    if key_path:
        for line in open(key_path).readlines():
            id = line.split()[0]
            sensekey = line.split()[1]  # multiple sense
            id_to_sensekey[id] = sensekey
    # data.xml 目录树的迭代器
    context = et.iterparse(data_path, tag='sentence')
    data = []
    poss = set() # 所有词性的集合
    for event, elem in context:
        sent_list = [] # 存储一个句子中所有的单词
        pos_list = [] # 存储一个句子中所有单词的词性
        for child in elem:
            word = child.get('lemma').lower()
            sent_list.append(word)
            pos = child.get('pos')
            pos_list.append(pos)
            poss.add(pos) # 存储所有句子中国所有单词的词性（set）

        i = -1
        for child in elem:
            # 对每个词进行分析
            if child.tag == 'wf':# 非target word
                i += 1
            elif child.tag == 'instance':# 若为targetword，则进行构造数据集
                i += 1
                """
                    <instance id="d000.s000.t000" lemma="long" pos="ADJ">long</instance>
                """
                id = child.get('id')
                lemma = child.get('lemma').lower()
                if '(' in lemma: # ??? no use
                    print (id)
                pos = child.get('pos')
                word = lemma + '#' + pos_dic[pos] # long#a
                if key_path:
                    sensekey = id_to_sensekey[id]
                else:
                    sensekey = None
                if is_training:
                    # 当为训练集时，且该词的词义数量只有一个的时候跳过，不必进行训练；筛选出词义大于1的训练数据
                    if word_count_info[word][0] <= 1 or word_count_info[word][1] <= 1:
                        continue
                # 一个句子中有多个target word，为每一个target word 准备上下文
                context = sent_list[:]
                if context[i] != lemma: # 有问题的情况抛出
                    print ('/'.join(context))
                    print (i)
                    print (lemma)
                # 把target word替换
                context[i] = '<target>'
                # 构造数据项
                x = {
                    'id': id, # d000.s000.t000
                    'context': context, # 一个句子中所有的单词，target word已替换
                    'target_sense': sensekey,  # don't support multiple answers #refer%2:32:01::
                    'target_word': word, # lemma: long#a
                    'poss': pos_list, # 所有单词的词性，与context一一对应
                    # 'sense_wn_fre' : sense_count_wn,
                    # 'sense_tr_fre' : sense_count_corpus,
                }

                data.append(x) # 每一个target 作为一个数据

    if is_training: # 为方便以后使用pos列表
        # 存储所有词性的集合
        poss_list = ['<pad>', '<eos>', '<unk>'] + list(sorted(poss))
        # print 'Wirting to tmp/pos_dic.pkl:' + ' '.join(poss_list)
        poss_map = dict(zip(poss_list, range(len(poss_list))))
        with open('../tmp/pos_dic.pkl', 'wb') as f:
            pickle.dump((poss_map), f)

    return data


def filter_word_and_sense(train_data, test_data, min_sense_freq=1, max_n_sense=40):
    """
    # 对train_data 筛选出与test_data 相交的target ,且筛选词义 target的词义出现的次数>min_sense_freq的target集合
    :param train_data:
    :param test_data:
    :param min_sense_freq: 太少的训练数据，得不到结果
    :param max_n_sense:
    :return: 筛选后的{target word : [语义1, 语义2...]}
    """

    """
    train_data数据形态
    {
    'id': 'd000.s000.t000', 
    'context': ['how', '<target>', 'have', 'it', 'be', 'since', 'you', 'review', 'the', 'objective', 'of', 'you', 'benefit', 'and', 'service', 'program', '?'], 
    'target_sense': 'long%3:00:02::', 
    'target_word': 'long#a', 
    'poss': ['ADV', 'ADJ', 'VERB', 'PRON', 'VERB', 'ADP', 'PRON', 'VERB', 'DET', 'NOUN', 'ADP', 'PRON', 'NOUN', 'CONJ', 'NOUN', 'NOUN', '.']
    }
    
    """
    train_words = set()
    # 从训练集中抽取所有的target word
    for elem in train_data:
        train_words.add(elem['target_word'])
    # 从测试集中抽取出所有的target word
    test_words = set()
    for elem in test_data:
        test_words.add(elem['target_word'])
    # 取交集导到 target word 的集合
    target_words = train_words & test_words
    #全量训练数据
    # target_words = train_words
    # 字典的子类，提供了可哈希对象的计数功能
    counter = collections.Counter()
    # 遍历训练集，取出
    for elem in train_data:
        # 筛选在target word 中的
        if elem['target_word'] in target_words:
            counter.update([elem['target_sense']]) # ['say%2:32:00::'] # 必须为可迭代对象

    # counter为 训练集中target_word出现的次数
    """
        {'say%2:32:00::': 1684,....}
    """
    # remove infrequent sense
    # 筛选出 出现的次数大于 min_sense_freq
    filtered_sense = [item for item in counter.items() if item[1] >= min_sense_freq]
    # [('say%2:32:00::', 1684)....] 从大到小排序
    count_pairs = sorted(filtered_sense, key=lambda x: -x[1])
    # senses [('say%2:32:00::',....)]
    # _ [(1684...)]
    senses, _ = list(zip(*count_pairs))
    # all_sense_to_id {'say%2:32:00::': 0, 'group%1:03:00::': 1....}
    all_sense_to_id = dict(zip(senses, range(len(senses))))
    # 存储target 和 词义的对应关系
    word_to_senses = {}
    """
    word_to_senses数据格式
        {'program#n': ['program%1:09:01::', 'program%1:09:00::', 'program%1:10:04::',}
    """
    for elem in train_data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        #从筛选后的词义中去选择
        if target_sense in all_sense_to_id:
            # 该target是否已经加入word_to_senses
            if target_word not in word_to_senses:
                # 未加入，新建一条key [value]
                word_to_senses.update({target_word: [target_sense]})
            else:
                # 已加入，则对value 表添加新的的词义
                if target_sense not in word_to_senses[target_word]:
                    word_to_senses[target_word].append(target_sense)

    # 存储最后的word_to_sense结果
    filtered_word_to_sense = {}
    for target_word, senses in word_to_senses.items():
        # 单个target 的sense 截取
        senses = sorted(senses, key=lambda s: all_sense_to_id[s])
        # 截取前max_n_sense几个词意
        senses = senses[:max_n_sense]
        # 词意小于1的词义筛掉
        if len(senses) > 1:  # must leave more than one sense
            # sense 乱序
            np.random.shuffle(senses)  # shuffle senses to avoid MFS
            filtered_word_to_sense[target_word] = senses
    """
    filtered_word_to_sense 格式
    {
        'program#n': ['program%1:09:01::', 'program%1:10:01::', 'program%1:09:00::', 'program%1:10:04::']
    }
    
    """
    return filtered_word_to_sense


def data_postprocessing(train_dataset, test_dataset, train_data, test_data, back_off_type="FS",
                        min_sense_freq=1, max_n_sense=40):
    """
    数据处理
    :param train_dataset: Train fileName
    :param test_dataset: Test fileName
    :param train_data: trian data
    :param test_data: tain data
    :param back_off_type: 策略类型
    :param min_sense_freq: 词义出现的最小的次数
    :param max_n_sense:  最多的词义
    :return:
        new_train_data, 筛选后的train_data
        new_test_data,  筛选后的test_data
        filtered_word_to_sense,   筛选后的target
        back_off_result,  back_off_result
        total,   FS模式下的 测试集中的target_sense  所有有效的target
        r ,FS模式 下 test_data中的target_sense ==  fs文件中的的id
    """
    # 对训练集的数据进行筛选：
    # 1. 与测试集的交集
    # 2. 每个语义出现次数 >= min_sense_freq
    # 3. target的语义条数 取前max_n_sense个
    filtered_word_to_sense = filter_word_and_sense(train_data, test_data, min_sense_freq, max_n_sense)
    """
       filtered_word_to_sense 格式
       {
           'program#n': ['program%1:09:01::', 'program%1:10:01::', 'program%1:09:00::', 'program%1:10:04::']
       }

       """
    # 根据筛选条件，构建新的train_data
    new_train_data = []
    for elem in train_data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        # 根据target 和 词义列表 筛选
        if target_word in filtered_word_to_sense and target_sense in filtered_word_to_sense[target_word]:
            new_train_data.append(elem)

    # 上同
    new_test_data = []
    for elem in test_data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        if target_word in filtered_word_to_sense and target_sense in filtered_word_to_sense[target_word]:
        # test will ignore sense not in train
            new_test_data.append(elem)
    #  '../data/All_Words_WSD/Output_Systems_ALL/MFS_semcor.key'
    mfs_key_path = _path.MFS_PATH.format(train_dataset)
    fs_key_path = _path.WNFS_PATH # '../data/All_Words_WSD/Output_Systems_ALL/WNFirstsense.key'

    # 封装mfs {"semeval2015.d003.s023.t004": people%1:14:00::}
    mfs_id_key_map = {}
    # MFS_semcor.key -> dict
    # 'senseval2.d000.s000.t000': 'art%1:06:00::'
    for line in open(mfs_key_path):
        id = line.split()[0]
        key = line.split()[1]
        mfs_id_key_map[id] = key

    fs_id_key_map = {}

    # WNFirsenss.key -》 dict
    # 'senseval2.d000.s000.t000': 'art%1:06:00::'
    for line in open(fs_key_path):
        id = line.split()[0]
        key = line.split()[1]
        fs_id_key_map[id] = key

    back_off_result = []

    mfs_using_fs_info = 0
    target_word_back_off = set()
    all_target_words = set()
    r = 0
    total = 0
    #对于测试数据集
    for elem in test_data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        # 存储所有的test数据集 中的target
        all_target_words.add(target_word)
        # 筛选
        if target_word not in filtered_word_to_sense:
            target_word_back_off.add(target_word)
            # 'senseval2' ！= ALL
            if test_dataset != _path.ALL_WORDS_TEST_DATASET[0]:  # ALL dataset id format has dataset name
                # 非ALL中 id = senseval2.d000.s000.t001
                id = test_dataset + '.' + elem['id']
            else:
                # ALL时 可以 id = d000.s000.t001
                id = elem['id']
            #  id 为 test_data 中的符合条件的id
            if back_off_type == 'FS':
                # [ ['d000.s000.t001', 'change_ringing%1:04:00::'] ]

                # new_test_data 数据集 与 fs 文件的重合部分
                back_off_result.append([elem['id'], fs_id_key_map[id]])  # fs_id_key_map[id] 可能是空的

                total += 1
                if target_sense == fs_id_key_map[id]:
                    # new_test_data 数据集的词义，与 fs词义相同

                    r += 1
            if back_off_type == 'MFS':  # dataset MFS may not cover all-words
                if id in mfs_id_key_map:
                    # new_test_data 数据集 与 mfs 文件的重合部分
                    back_off_result.append([elem['id'], mfs_id_key_map[id]])
                else:
                    # 未找到；把Fs文件中的拿过来，这说明fs文件的值非常全。

                    mfs_using_fs_info += 1 # new_test_data 数据集 不与 mfs 文件的重合的数量
                    back_off_result.append([elem['id'], fs_id_key_map[id]])
    # back_off_result ：{[['d000.s000.t001', 'change_ringing%1:04:00::']]}

    print('***MFS Using wordnet information instance number:%d ' % (mfs_using_fs_info))
    print('***Using back off target words: %s/%s' % (len(target_word_back_off), len(all_target_words)))
    print('backoff acc = %s' % (r/total))
    # ../tmp/back_off_results-{FS、FMS}.txt
    back_off_result_path = _path.BACK_OFF_RESULT_PATH.format(back_off_type)
    print('***Writing to back_off_results to file:%s' % back_off_result_path)
    with open(back_off_result_path, 'w') as f:
        for instance_id, predicted_sense in back_off_result:
            f.write('%s %s\n' % (instance_id, predicted_sense))

    return new_train_data, new_test_data, filtered_word_to_sense, back_off_result, total, r


def data_postprocessing_for_validation(val_data, filtered_word_to_sense=None):
    """

    :param val_data: val data数据集
    :param filtered_word_to_sense: 筛选后的targt sense
    :return: 筛选后的val_data
    """
    #
    new_val_data = []
    for elem in val_data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        if target_word in filtered_word_to_sense and target_sense in filtered_word_to_sense[target_word]:
            new_val_data.append(elem)
    return new_val_data


def test_data_postprocessing(train_dataset, train_target_words, test_data, back_off_type='MFS'):
    key_path = None
    if back_off_type == 'MFS':
        key_path = _path.MFS_PATH.format(train_dataset)
    elif back_off_type == 'FS':
        key_path = _path.WNFS_PATH

    id_key_map = {}
    if key_path:
        for line in open(key_path):
            id = line.split()[0]
            key = line.split()[1]
            id_key_map[id] = key

    back_off_result = []
    new_test_data = []
    for d in test_data:
        if d['target_word'] in train_target_words:
            new_test_data.append(d)
        else:
            id = d['id']
            if id in id_key_map:
                back_off_result.append([id, id_key_map[id]])
    return new_test_data, back_off_result


def build_vocab(data):
    """
    :param data: new train data
    :return: a dict with words as key and ids as value
    """
    counter = collections.Counter()
    # 对于train data 中的word 进行计数
    for elem in data:
        #
        counter.update(elem['context'])
        counter.update([elem['target_word']])
        # counter.update([elem['']])
    """
        counter ({'how':1479,'<target>':62373})
    """
    # remove infrequent words/context
    # 去掉只出现过一次的word
    min_freq = 1
    filtered = [item for item in counter.items() if item[1] >= min_freq]
    # 大到小排序
    count_pairs = sorted(filtered, key=lambda x: -x[1])
    # 把所有的word集合存入words
    words, _ = list(zip(*count_pairs))
    # 加入这三个单词
    add_words = ['<pad>', '<eos>', '<unk>'] + list(words)
    # 按序生成dict
    word_to_id = dict(zip(add_words, range(len(add_words))))

    return word_to_id


def build_sense_ids(word_to_senses):
    """

    :param word_to_senses:
    :return: target_word_to_id
    """
    words = list(word_to_senses.keys())
    # 不能保证前后顺序
    target_word_to_id = dict(zip(words, range(len(words))))
    # 不能保证前后顺序
    target_sense_to_id = [dict(zip(word_to_senses[word], range(len(word_to_senses[word])))) for word in words]

    n_senses_from_word_id = dict([(target_word_to_id[word], len(word_to_senses[word])) for word in words])
    return target_word_to_id, target_sense_to_id, n_senses_from_word_id, word_to_senses


class Instance:
    pass


def convert_to_numeric(data, word_to_id, target_word_to_id, target_sense_to_id,
                       ignore_sense_not_in_train=True, mode=''):
    """

    :param data:
        {'id': 'd000.s000.t005', 'context': ['how', 'long', 'have', 'it', 'be', 'since', 'you', 'review', 'the', 'objective', 'of', 'you', 'benefit', 'and', '<target>', 'program', '?'], 'target_sense': 'service%1:04:07::', 'target_word': 'service#n', 'poss': ['ADV', 'ADJ', 'VERB', 'PRON', 'VERB', 'ADP', 'PRON', 'VERB', 'DET', 'NOUN', 'ADP', 'PRON', 'NOUN', 'CONJ', 'NOUN', 'NOUN', '.']},
    :param word_to_id:
        {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'the': 3, ',': 4, '<target>': 5, '.': 6, 'of': 7, }
    :param target_word_to_id:
        {'service#n': 0, 'program#n': 1, 'permit#v': 2, 'become#v': 3, 'rather#r': 4, 'morale#n': 5}
    :param target_sense_to_id:
        [{'service%1:14:05::': 0, 'service%1:04:00::': 1, 'service%1:04:08::': 2, 'service%1:14:00::': 3, 'service%1:04:01::': 4, 'service%1:04:07::': 5}, {'program%1:10:01::': 0, 'program%1:09:00::': 1, 'program%1:10:04::': 2, 'program%1:09:01::': 3},]
    :param ignore_sense_not_in_train:
    :param mode:
    :return:
    """
    words_notin_vocab = []
    # 加载pos_to_id
    with open('../tmp/pos_dic.pkl', 'rb') as f:
        pos_to_id = pickle.load(f)
    """
        {'<pad>': 0, '<eos>': 1, '<unk>': 2, '.': 3, 'ADJ': 4, 'ADP': 5, 'ADV': 6, 'CONJ': 7, 'DET': 8, 'NOUN': 9, 'NUM': 10, 'PRON': 11, 'PRT': 12, 'VERB': 13, 'X': 14}
    """
    all_data = []
    target_tag_id = word_to_id['<target>']
    instance_sensekey_not_in_train = []
    for insi, instance in enumerate(data):
        words = instance['context']
        poss = instance['poss']
        # 确保word 与 pos 一一对应
        assert len(poss) == len(words)
        ctx_ints = []
        pos_ints = []
        # 对一条train_data 中的上下文word
        for i, word in enumerate(words):
            # 单词和词性对应
            if word in word_to_id:
                ctx_ints.append(word_to_id[word])
                pos_ints.append(pos_to_id[poss[i]])
            elif len(word) > 0:
                # 未在单词列表里，则用<unk>
                ctx_ints.append(word_to_id['<unk>'])
                pos_ints.append(pos_to_id['<unk>'])
                # 记录该单词
                words_notin_vocab.append(word)

        stop_idx = ctx_ints.index(target_tag_id)
        xf = np.array(ctx_ints[:stop_idx], dtype=np.int32)
        pf = np.array(pos_ints[:stop_idx], dtype=np.int32)
        # 最远的单词在第一个位置。
        xb = np.array(ctx_ints[stop_idx + 1:], dtype=np.int32)[::-1]
        pb = np.array(pos_ints[stop_idx + 1:], dtype=np.int32)[::-1]

        instance_id = instance['id'] # d00.s00.t005
        target_word = instance['target_word'] # long#v
        target_sense = instance['target_sense'] # 'service%1:04:07::'
        # sense_count_wn = instance['sense_wn_fre']
        # sense_count_tr = instance['sense_tr_fre']

        try:
            target_id = target_word_to_id[target_word] # 7
            senses = target_sense_to_id[target_id] #{service%1:04:07:2, ***:8}
        except KeyError as e:
            # print e
            continue

        if target_sense in senses:  # test will ignore sense not in train, same as data_postprocessing
            sense_id = senses[target_sense] # 3
        else:
            instance_sensekey_not_in_train.append([instance_id, target_sense])
            if ignore_sense_not_in_train:
                continue
            else:
                sense_id = 0  # sensekey not in train is classified as label 0

        instance = Instance()
        instance.id = instance_id
        # instance.sense_tr_fre = sense_count_tr
        # instance.sense_wn_fre = sense_count_wn
        # context f:forward b:back
        instance.xf = xf
        instance.xb = xb
        # pos
        instance.pf = pf
        instance.pb = pb
        # ???why context <target>:1

        # instanc
        # instance.target_word_id = word_to_id['<unk>']  #这里进行了改动
        t = target_word.split("#")[0]
        instance.target_word_id = word_to_id[t] if t in word_to_id else word_to_id['<unk>'] #这里进行了改动
        # target_pos_id  'ADJ': 4
        instance.target_pos_id = pos_ints[stop_idx]

        instance.target_id = target_id # hava#v ：1
        instance.sense_id = sense_id #  service%1:14:05::': 0

        all_data.append(instance)

    tmp_lenth = len(instance_sensekey_not_in_train)
    if tmp_lenth:
        print('###%s instance_sensekey_not_in_train: %s' % (mode, tmp_lenth))

    store_notin_vocab_words(words_notin_vocab, mode=mode)
    print('%s words_notin_vocab:%d' % (mode, len(words_notin_vocab)))

    return all_data


def store_notin_vocab_words(words_notin_vocab, mode='', clean=True):
    """

    :param words_notin_vocab:不在词表的单词
    :param mode: train/test/val
    :param clean: 是否追加
    :return:
    """
    if clean:
        old = []
    else:
        try:
            old = open(path_words_notin_vocab.format(mode)).read()
            old = old.split('\n')
        except Exception as e:
            old = []

    ws = []
    for word in words_notin_vocab:
        try:
            word = word.decode('utf-8')
            ws.append(word)
        except Exception as e:
            continue
    new = set(ws + old)
    open(path_words_notin_vocab.format(mode), 'w').write('\n'.join(new))

def batch_generator_by_sama_target_word(is_training, max_batch_size, data):
    same_target_word_dict = {}
    for instance in data:
        target_id = instance.target_id
        if target_id not in same_target_word_dict:
            same_target_word_dict[target_id] = []
            same_target_word_dict[target_id].append(instance)
        else:
            same_target_word_dict[target_id].append(instance)

    # print("batch_generation_target_word_num:",len(same_target_word_dict))


    #平均batch
    # temp_list = []
    # for target_id, instance_list in same_target_word_dict.items():
    #     temp_list.extend(instance_list)
    # data_len = len(temp_list)
    #
    # # 拆分list
    # if is_training:
    #     random.shuffle(temp_list)
    # same_target_word_list = [temp_list[i:i+(max_batch_size)] for i in range(0, len(temp_list), max_batch_size)]
    # # same_target_word_list += tmp

    #不平均batch
    same_target_word_list = []
    data_len = 0
    for target_id, instance_list in same_target_word_dict.items():
        data_len += len(instance_list)
        if len(instance_list) <= max_batch_size:
            same_target_word_list.append(instance_list)
        else:
            # 拆分list
            tmp = [instance_list[i:i+max_batch_size] for i in range(0, len(instance_list), max_batch_size)]
            same_target_word_list += tmp
    return  same_target_word_list ,data_len

def batch_generator(is_training, batch_size, max_gloss_words, data, dict_data, pad_id, n_step_f, n_step_b, pad_last_batch=False):
    """

    :param is_training: true or false
    :param batch_size: 8
    :param max_gloss_words:  100
    :param data: train_data
    :param dict_data: [3dim,,2dim,3dim]
    :param pad_id: word_to_id['<pad>']
    :param n_step_f:  30  # forward context length
    :param n_step_b: 30   # backward context length
    :param pad_last_batch:  true  or false
    :return:
    """

    # same_target_word_list,data_len = batch_generator_by_sama_target_word(is_training, batch_size, data)
    # for i in range(len(same_target_word_list)):
    #     batch = same_target_word_list[i]
    # 切分batch数据


    data_len = len(data)
    n_batches_float = data_len / float(batch_size)
    n_batches = int(math.ceil(n_batches_float)) if pad_last_batch else int(n_batches_float)
    if is_training:
        random.shuffle(data)
    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]
        batch_size = len(batch)

        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)
        forward_text = np.zeros([batch_size, n_step_f], dtype=np.int32)
        backward_text = np.zeros([batch_size, n_step_b], dtype=np.int32)
        forward_pos = np.zeros([batch_size, n_step_f], dtype=np.int32)
        backward_pos = np.zeros([batch_size, n_step_b], dtype=np.int32)

        forward_text.fill(pad_id)
        backward_text.fill(pad_id)
        forward_pos.fill(pad_id)
        backward_pos.fill(pad_id)
        xfs.fill(pad_id)
        xbs.fill(pad_id)
        xfbs.fill(pad_id)

        # context pos
        # pfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        # pbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        # pfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)  # 0 is pad for pos, no need pad_id

        # id
        # batch的数据中所有的id
        instance_ids = [inst.id for inst in batch]


        # labels
        target_ids = [inst.target_id for inst in batch]
        # print("target_ids")
        # print(target_ids)
        sense_ids = [inst.sense_id for inst in batch]

        if len(target_ids) < batch_size:  # padding
            n_pad = batch_size - len(target_ids)
            # print('Batch padding size: %d'%(n_pad))
            target_ids += [0] * n_pad
            sense_ids += [-1] * n_pad #####
            instance_ids += [0] * n_pad  # instance_ids += [''] * n_pad
        # 【target_id,...】
        target_ids = np.array(target_ids, dtype=np.int32)
        sense_ids = np.array(sense_ids, dtype=np.int32)
        # 【（10，100）】
        # print("target_ids")
        # print(target_ids)
        glosses_ids = [dict_data[0][target_ids[i]] for i in range(batch_size)]
        # print(glosses_ids)
        # print(glosses_ids[0].shape)
        # [(10,)...]
        # glosses_lenth = [dict_data[1][target_ids[i]] for i in range(batch_size)]
        #
        sense_mask = [dict_data[2][target_ids[i]] for i in range(batch_size)]

        # x forward backward
        for j in range(batch_size):
            #不paddind的时候给值
            if i * batch_size + j < data_len:
                # 规定上下文的上限
                n_to_use_f = min(n_step_f-1, len(batch[j].xf)) #
                n_to_use_b = min(n_step_b-1, len(batch[j].xb)) #
                if n_to_use_f:
                    # 取离target word 较近的上下文
                    # 前置 pad
                    xfs[j, -n_to_use_f:] = batch[j].xf[-n_to_use_f:]
                    # pfs[j, -n_to_use_f:] = batch[j].pf[-n_to_use_f:]
                    # 后置 pad
                    forward_text[j, :n_to_use_f] = batch[j].xf[-n_to_use_f:]
                    forward_pos[j, :n_to_use_f] = batch[j].pf[-n_to_use_f:]
                    forward_text[j][n_to_use_f] = batch[j].target_word_id # 加入了<unk> / target word
                    forward_pos[j][n_to_use_f] = batch[j].target_pos_id
                if n_to_use_b:#上同
                    xbs[j, -n_to_use_b:] = batch[j].xb[-n_to_use_b:]
                    # pbs[j, -n_to_use_b:] = batch[j].pb[-n_to_use_b:]
                    backward_text[j, :n_to_use_b] = batch[j].xb[-n_to_use_b:]
                    backward_pos[j, :n_to_use_b] = batch[j].pb[-n_to_use_b:]
                    backward_text[j][n_to_use_b] = batch[j].target_word_id
                    backward_pos[j][n_to_use_b] = batch[j].target_pos_id
                # 将上文 + <unk> + 下文 拼接在一起（id）
                # xfbs[j] = np.concatenate((xfs[j], [batch[j].target_word_id], xbs[j]), axis=0)
                # pfbs[j] = np.concatenate((pfs[j], [batch[j].target_pos_id], pbs[j]), axis=0)

        forward_text = torch.from_numpy(forward_text).to(torch.int64)
        backward_text = torch.from_numpy(backward_text).to(torch.int64)
        forward_pos = torch.from_numpy(forward_pos).to(torch.int64)
        backward_pos = torch.from_numpy(backward_pos).to(torch.int64)
        ## no use
        # ids = torch.tensor(instance_ids) # 新增
        # xfbs = torch.from_numpy(xfbs).to(torch.int64)
        # pfs = torch.from_numpy(pfs).to(torch.int64)
        # pbs = torch.from_numpy(pbs).to(torch.int64)
        # pfbs = torch.from_numpy(pfbs).to(torch.int64)
        # target_ids = torch.from_numpy(target_ids)#在target_word_list中的位置
        sense_ids = torch.from_numpy(sense_ids)
        # glosses_lenth = torch.tensor(glosses_lenth).to(torch.int64)
        glosses_ids = torch.tensor(glosses_ids).to(torch.int64)
        sense_mask = torch.tensor(sense_mask).to(torch.int64)

        # 新增id ,为方便获取原始文件
        yield (forward_text, backward_text, sense_ids, glosses_ids, sense_mask, forward_pos, backward_pos)
        # yield ([target_id, word_to_sense_id, forward_text, backward_text, glosses_ids, forward_pos, backward_pos], ids)


# get gloss from dictionary.xmltrain_dataset, target_word_to_id
def load_dictionary(dataset, target_words=None, expand_type=0, n_hyper=3, n_hypo=3):
    """

    :param dataset:train_dataset
    :param target_words:target_word_to_id
    :param expand_type:
    :param n_hyper:
    :param n_hypo:
    :return:
    """
    gloss_dic = {}
    if dataset in _path.LS_DATASET: # no use
        dic_path = _path.LS_DIC_PATH.format(dataset)
        target_words = None  # Lexical task don't need target words filter
    elif dataset in _path.ALL_WORDS_TRAIN_DATASET: # selected
        dic_path = _path.ALL_WORDS_DIC_PATH.format(dataset)
    else:
        raise ValueError(
            '%s or %s. Provided: %s' % (','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TRAIN_DATASET), dataset))
    # 解析dict.xml 文件
    soup = BeautifulSoup(open(dic_path), 'lxml')
    all_sense_tag = soup.find_all('sense')
    for sense_tag in all_sense_tag:
        # 词义的唯一标识
        id = sense_tag['id'] # 'newspaper%1:06:00::'
        key = id  # all-words
        # key = id.replace("-", "'")  # senseval2_LS README EG:pull_in_one-s_horns%2:32:00::
        gloss = sense_tag['gloss'] # "'the physical object that is the product of a newspaper publisher'"
        if target_words:  # for all_words task
            # have#v
            target_word = sense_tag.parent['item']
            # 筛选targetword
            if target_word in target_words:
                gloss_dic[id] = gloss
        else:  # for lexical example task
            gloss_dic[id] = gloss
    return gloss_dic


def split_sentence(sent):
    # 句子预处理
    sent = re.findall(r"[\w]+|[^\s\w]", sent)
    for i, word in enumerate(sent):
        # 词型还原 runing -> run
        # sent[i] = wordnet_lemmatizer.lemmatize(word)
        sent[i] = word
    return sent


# make initial sense id(in dataset) to new sense id, and make numeric for gloss defination
def bulid_dictionary_id(gloss_dict, target_sense_to_id, word_to_id, pad_id, mask_size, max_gloss_words=100):
    """

    :param gloss_dict:
        {'newspaper%1:06:00::': 'the physical object that is the product of a newspaper publisher', 'newspaper%1:14:00::': 'a business firm that publishes newspapers'}
    :param target_sense_to_id:
    :param word_to_id:
    :param pad_id:
    :param mask_size:
    :param max_gloss_words:
    :return:
    """

    # t_max_gloss_words = max([len(split_sentence(g)) for g in gloss_dict.values()])
    # print('original max_gloss_words: %s' % (t_max_gloss_words))
    # target word num
    n_target_words = len(target_sense_to_id) # n_target_words = 679
    print('n_target_words: %s' % n_target_words)
    # 词义数量的最大值
    max_n_sense = max([len(sense_to_ids) for sense_to_ids in target_sense_to_id])
    print('max_n_sense %d' % (max_n_sense))
    # shape (679, 10, 100)
    gloss_to_id = np.zeros([n_target_words, max_n_sense, max_gloss_words], dtype=np.int32)
    gloss_to_id.fill(pad_id)

    words_notin_vocab = []

    gloss_lenth = np.zeros([n_target_words, max_n_sense], dtype=np.int32)
    sense_mask = np.zeros([n_target_words, max_n_sense], dtype=np.int32)
    for i, sense_to_ids in enumerate(target_sense_to_id):
        if i % 500 == 0:
            print("Bulid dictionary: %s/%s" % (i, len(target_sense_to_id)))
        for id0 in sense_to_ids:  # id0 is the initial id in dataset
            j = sense_to_ids[id0]
            # gloss 单词列表（已经过词型还原）
            gloss_words = split_sentence(gloss_dict[id0])
            # 第i个target word, 第j个词义，
            sense_mask[i][j] = 1
            words = []
            # 对gloss 进行 与词汇表的对比，存在即表示对应的成id，未存在保存在words_notion_vacab

            for word in gloss_words:
                if word in word_to_id:
                    words.append(word_to_id[word])
                elif len(word) > 0:
                    words.append(word_to_id['<unk>'])
                    words_notin_vocab.append(word)
            # 防止gloss 过长，截取取n个词
            words = words[:max_gloss_words]

            if len(words) > 0:
                # how?
                gloss_to_id[i, j, :len(words)] = words  # pad in the end
                gloss_lenth[i][j] = len(words)
    # 把gloss中存在的单词，但是未在我们定义的词汇表中的词汇存储下来
    store_notin_vocab_words(words_notin_vocab, mode='gloss')
    print('%s words_notin_vocab:%d' % ('gloss', len(words_notin_vocab)))
    return [gloss_to_id, gloss_lenth, sense_mask], max_n_sense

import os
def create_auxiliary_metrix():
    # basedir = os.path.dirname(__file__)
    # print("basedir:" + basedir)
    current_path = os.path.dirname(os.path.abspath(__file__))
    p = 'include_target.json'
    """ 
    {"target_word": "newspaper", 
    "sense": "newspaper%1:06:00::", 
    "gloss": "the physical ", 
    "include_target": ["newspaper", "physical"]}
    """
    file = open(os.path.join(current_path, p), "r")
    sense_dict = {}   # key:value | newspaper%1:06:00:: : 1 - x
    gloss_dict = {}   # "gloss context" : 1-2701?
    gloss_frequence_dict = {} # "gloss context" : 1 or 2 or 3
    word_dict = {} # target_word列表，值1-653  newspaper:1-653
    include_dict = {} # "gloss context" : ["newspaper", "physical"]
    sense_to_gloss = {} # ewspaper%1:06:00:: : "gloss context"
    word_to_sense = {} # “newspaper”
    for line in file:
        temp_dict = json.loads(line)
        target_word = temp_dict['target_word']
        sense = temp_dict['sense']
        gloss = temp_dict['gloss']
        include_target = temp_dict['include_target']

        if target_word not in word_dict:
            # 值 1 - 653 | key:value |  newspaper:1
            word_dict[target_word] = len(word_dict) + 1
        else:
            pass
        if target_word not in word_to_sense:
            word_to_sense[target_word] = []
            if sense not in word_to_sense[target_word]:
                word_to_sense[target_word].append(sense)
        else:
            if sense not in word_to_sense[target_word]:
                word_to_sense[target_word].append(sense)

        if sense not in sense_dict:
            # key:value | newspaper%1:06:00:: : 1-
            # gloss:1-2700
            sense_dict[sense] = len(sense_dict) + 1
            sense_to_gloss[sense] = gloss
        else:
            pass

        if gloss not in gloss_dict:
            # "gloss context" : 1-2701?
            gloss_dict[gloss] = len(gloss_dict) + 1
            include_dict[gloss] = include_target
        else:
            # gloss 有相同的，那么include_target 就不一样了吧？ === 应该是一样的，一般不会解释自己，然后再***
            pass


        #记录该词义文本出现多少次
        if gloss not in gloss_frequence_dict:
            gloss_frequence_dict[gloss] = 1
        else:
            gloss_frequence_dict[gloss] += 1


    sense_to_gloss_id = torch.zeros(len(sense_dict),dtype=torch.int64)  # (2979) 0pad 1-2700表示gloss
    word_to_sense_id = torch.zeros((len(word_dict), 10),dtype=torch.int64)  # (654,10) 0-pad 1-2971表示sense

    # print(gloss_frequence_dict)
    gloss_frequence = torch.zeros((len(gloss_dict)),dtype=torch.float) #(2700)
    # sense_frequence = torch.zeros((len(gloss_dict), 6, 10),dtype=torch.int64)
    gloss_to_word_id = torch.zeros((len(gloss_dict), 6),dtype=torch.int64)  # (2700,6)  0pad 1-654表示word
    gloss_to_word_mask = torch.zeros((len(gloss_dict), 100), dtype=torch.int64)  # (2700,100) 1-6表示target_word_position, 0表示non_target
    sense_mask = torch.zeros((len(gloss_dict), 6, 10), dtype=torch.float)  # (2700,6,10) 1表示有sense,0表示没有sense

    n = 0
    for i in gloss_dict:
        gloss_frequence[n] = gloss_frequence_dict[i]
        n += 1

    n = 0 #2979
    for i in sense_dict:
        sense_to_gloss_id[n] = gloss_dict[sense_to_gloss[i]]
        n += 1

    n = 0 #2700
    for i in word_dict:
        m = 0 #10
        for j in word_to_sense[i]:
            if m < 10:
                word_to_sense_id[n][m] = sense_dict[j]
                m += 1
            else:
                break
        n += 1

    n = 0#2700
    for i in gloss_dict:
        m = 0 #6
        index = 0 #100
        for w in i.split(' '):
            if w in word_dict and index<100:
                gloss_to_word_id[n][m] = word_dict[w]
                gloss_to_word_mask[n][index] = m+1
                m += 1
            index += 1
        n += 1

    for i in range(2700):
        for j in range(6):
            if gloss_to_word_id[i][j] > 0:
                for k in range(10):
                    if word_to_sense_id[gloss_to_word_id[i][j]-1][k] > 0:
                        sense_mask[i][j][k] = 1.0

    init_alpha = torch.zeros_like(sense_mask)
    ###均匀分布初始化alpha
    # mask_sum = torch.sum(sense_mask, dim=-1, dtype=torch.float)
    # for i in range(2700):
    #     for j in range(6):
    #         for k in range(10):
    #             if sense_mask[i][j][k] != 0:
    #                 init_alpha[i][j][k] = 1/mask_sum[i][j]

    ###按照词频初始化alpha
    g = torch.cat((torch.zeros(1), gloss_frequence), dim=0)  # (2701, HDsize)
    # 得到sense表征
    # print(g)
    a = torch.gather(g, 0, sense_to_gloss_id)  # (2979)
    a = torch.cat((torch.zeros(1), a), dim=0)  # (2980)
    # 得到target_word表征
    b = torch.gather(a.reshape(-1, 1).expand(-1, 10), 0, word_to_sense_id.reshape(654, -1).expand(-1, -1))  # (654, 10)
    b = torch.cat((torch.zeros(1, config.max_n_sense), b), dim=0)  # (655, 10)
    # 得到指定gloss中的target_word表征
    c = torch.gather(b.reshape(655, 1, 10).expand(-1, 6, -1), 0, gloss_to_word_id.reshape(2700, -1, 1).expand(-1, -1, 10))  # (2700, 6, 10)
    sum_fre = torch.sum(c, dim=-1)
    init_alpha = torch.where(sense_mask != 0, c/sum_fre.unsqueeze(-1).expand(-1,-1,10), sense_mask)

    return [sense_to_gloss_id, word_to_sense_id, gloss_to_word_id, gloss_to_word_mask, sense_mask, init_alpha], gloss_dict

def gloss_to_id(gloss_dict, word_to_id):#####
    gloss_id = torch.zeros((2700, 100),dtype=torch.int64).to("cuda")
    n = 0
    for i in gloss_dict:
        t = split_sentence(i)
        words = []
        # 对gloss 进行 与词汇表的对比，存在即表示对应的成id，未存在保存在words_notion_vacab

        for word in t:
            if word in word_to_id:
                words.append(word_to_id[word])
            elif len(word) > 0:
                words.append(word_to_id['<unk>'])
        # 防止gloss 过长，截取取n个词
        words = words[:100]

        if len(words) > 0:
            gloss_id[n][:len(words)] = torch.tensor(words)  # pad in the end
        n += 1
    return gloss_id



if __name__ == '__main__':
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
    # gloss_id = gloss_to_id(gloss_dict, word_to_id)  # (2700,100)