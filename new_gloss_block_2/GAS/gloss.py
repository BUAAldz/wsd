import numpy as np
import re
import torch
import torch.nn as nn

def split_sentence(sent):
    # 句子预处理
    sent = re.findall(r"[\w]+|[^\s\w]", sent)
    for i, word in enumerate(sent):
        sent[i] = word
    return sent

def create_gloss(word_to_id, glove_init_emb):
    f = open("all_gloss.txt", 'r')
    gloss_to_id = np.zeros([2700, 100], dtype=np.int32)
    gloss_to_id.fill(0)

    n = 0
    for i in f:
        gloss_words = split_sentence(i.strip())
        words = []
        for word in gloss_words:
            if word in word_to_id:
                words.append(word_to_id[word])
            elif len(word) > 0:
                words.append(word_to_id['<unk>'])
        # 防止gloss 过长，截取取n个词
        words = words[:100]

        if len(words) > 0:
            gloss_to_id[n, :len(words)] = words  # pad in the end
            n += 1

    gti = torch.from_numpy(gloss_to_id).to(torch.int64)
    glove_embedding = nn.Embedding.from_pretrained(glove_init_emb)
    glosses_emb = glove_embedding(gti) #(2700,100,300)

    tmp = torch.zeros(2700, 300)
    for i in range(2700):
        t = torch.zeros(300)
        m = 0
        for j in range(100):
            if torch.sum(glosses_emb[i][j])!=0:
                t += glosses_emb[i][j]
                m += 1
        t = t/m
        tmp[i] = t

    temp = tmp.numpy()
    np.save("./gloss.npy", temp)
