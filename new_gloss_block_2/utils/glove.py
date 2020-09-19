# -*- coding: utf-8 -*-
"""
 @version: python2.7
 @author: luofuli
 @time: 2017/7/1
"""

import numpy as np
from utils import path
import torch
_path = path.WSD_path()
data_dir = _path.DATA_DIR


def load_glove(path=None, dim=None, size=None):   # download from https://nlp.stanford.edu/projects/glove/
    """

    :param path:
    :param dim:
    :param size:
    :return: word2vec
    """
    if path is None:
        if size=='6B':
            path = data_dir + 'glove.6B.' + str(dim) + 'd.txt'
        elif size=='42B' and dim==300: # selected
            path = data_dir+'glove.42B.300d.txt'
        elif size=='840B' and dim==300:
            path = data_dir+'glove.840B.300d.txt'
        else:
            print(u'No pre-trained word-embeddings at dir:%s' % (data_dir))
            exit(-3)

    wordvecs = {}

    # 为避免调试内存溢出
    # TODO：为避免调试内存溢出，正式运行记得删除
    top_10000 = 100
    i = 0

    with open(path, 'r',encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:

            i+=1
            if (i >= top_10000):
                print("word2vec select top %d vct" % (top_10000))
                break

            tokens = line.split(' ')
            # 第一个是单词 不需要
            vec = np.array(tokens[1:], dtype=np.float32)
            wordvecs[tokens[0]] = vec

    return wordvecs


def fill_with_gloves(word_to_id, path=None, emb_size=None, vocab_size=None, wordvecs=None):
    """

    :param word_to_id: word_to_id,
    :param path: _path.GLOVE_VECTOR
    :param emb_size: 300
    :param vocab_size: '42B'
    :param wordvecs:
    :return:
    """
    if not wordvecs:
        # 加载词向量
        wordvecs = load_glove(path, emb_size, vocab_size)
    # 单词数 28858 训练集筛选后的所有的单词
    n_words = len(word_to_id)

    if emb_size is None:
        # embedding 的 dim
        emb_size = len(wordvecs[wordvecs.keys()[0]])
    #
    res = np.zeros([n_words, emb_size], dtype=np.float32)
    n_not_found = 0
    words_notin = set()
    for word, id in word_to_id.items():
        if '#' in word: #  useless 无词性标记
            word = word.split('#')[0]   # Remove pos tag
        # 合成单词拆分
        if '-' in word:
            words = word.split('-')
        elif '_' in word:
            words = word.split('_')
        else:
            words = [word]

        vecs = []
        for w in words:
            if w in wordvecs:
                vecs.append(wordvecs[w])    # add word2vec for multi-word
        if vecs != []:
            # 合成词，求平均的embedding 也太秀了
            res[id, :] = np.mean(np.array(vecs), 0)
        else:
            # 词向量中不存在的单词，记录
            words_notin.add(word)
            n_not_found += 1
            # 正则化随机生成不存在的单词
            res[id, :] = np.random.normal(0.0, 0.1, emb_size)
    print ('n words not found in glove word vectors: ' + str(n_not_found))
    # open('../tmp/word_not_in_glove.txt','w').write((u'\n'.join(words_notin)).encode('utf-8'))
    # 转成tensor
    res = torch.from_numpy(res)

    return res

def random_init(word_to_id, emb_size = None):
    n_words = len(word_to_id)
    res = torch.randn((n_words, emb_size), dtype=torch.float)

    return res