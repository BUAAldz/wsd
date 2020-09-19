import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy  as np

# 定义损失函数
class Marginal_loss(nn.Module):
    def __init__(self, eps, config):
        super(Marginal_loss, self).__init__()
        self.eps = eps # 2
        self.config = config

    def forward(self, sentence, all_gloss, sense_mask, sense_ids, alpha, mode):
        """
        :param sentence: (8,300) 每个句子经过lstm的向量化表示
        :param all_gloss: （8,10,300）
        :param sense_mask: （8,10） 第三位累加到第二维了 1024 or 0
        :param sense_ids:（8，）
        :param alpha (2700,6,10)
        :return:
        """
        # 求正例与原句的l2距离
        # target_gloss (8,300)
        batch_size = sentence.shape[0]
        target_gloss = torch.ones(batch_size, self.config.LSTMHD, device='cuda')
        for i in range(batch_size):
            try:
                target_gloss[i] = all_gloss[i][sense_ids[i]]
            except:
                print("i:",i)
                print("all_gloss.shape:",all_gloss.shape)
                print("sense_ids.shape:",sense_ids.shape)
                print("sentence.shape", sentence.shape)
                # print("sense_ids[i]", sense_ids[i])
        pos = F.pairwise_distance(sentence, target_gloss)


        #随机选取负例
        # neg_index = []
        # for i in range(self.config.batch_size):
        #     index = random.randint(0, sense_mask[i] - 1)
        #     while(index == sense_ids[i]):                     #如果随机选择的词义恰好是正例，则换一个
        #         index = random.randint(0, sense_mask[i] - 1)
        #     neg_index.append(index)

        # 选择最正的负例
        random_neg = torch.ones(batch_size, self.config.LSTMHD, device='cuda')
        for i in range(batch_size):
            single_sentence = torch.ones(1, self.config.LSTMHD, device='cuda')
            single_sentence[0] = sentence[i][:]
            single_sentence_muti_sence_distence = []
            for j in range(all_gloss.size(1)):
                single_sence_gloss = torch.ones(1, self.config.LSTMHD, device='cuda') * 10
                single_sence_gloss[0] = all_gloss[i][j][:]
                single_distence = F.pairwise_distance(single_sentence,single_sence_gloss).item()
                # print(single_distence)

                if sense_mask[i][j].item() != 0: # 若是空词义，则距离最大
                    if sense_ids[i].item() != j: # 除掉正例，计算剩余负例的距离
                        single_sentence_muti_sence_distence.append(single_distence)
                    else:
                        single_sentence_muti_sence_distence.append(10)
                else:
                    single_sentence_muti_sence_distence.append(10)
            # 获得最小值的index
            min_index = single_sentence_muti_sence_distence.index(min(single_sentence_muti_sence_distence))
            random_neg[i] = all_gloss[i][min_index]

        #求负例的与原句的l2距离 （8,）
        neg = F.pairwise_distance(sentence, random_neg)

        z = torch.zeros(batch_size).to("cuda")
        # 计算marginal_loss
        if mode == "train":

            # 小于0的用0补充
            a = alpha.reshape(2700 * 6, -1)  # (2700*6,10)
            b = torch.ones_like(a)
            mask = torch.sum(a, dim=-1).reshape(-1, 1).expand(-1, 10)  # (2700， 6)
            tmp = torch.where(mask == 0, b, a)  # (2700*6,10)
            # print(tmp[0])
            # print(torch.abs(torch.max(tmp, dim=1)[0]-1))
            # print(torch.max(pos - neg + self.eps, z))
            # print(torch.abs(torch.max(tmp, dim=1)[0]-1).shape)
            loss = torch.sum(torch.max(pos - neg + self.eps, z))*0.875 + torch.sum(torch.abs(torch.max(tmp, dim=1)[0]-1))*0.125
        else:
            a = alpha.reshape(2700 * 6, -1)  # (2700*6,10)
            b = torch.ones_like(a)
            mask = torch.sum(a, dim=-1).reshape(-1, 1).expand(-1, 10)  # (2700， 6)
            tmp = torch.where(mask == 0, b, a)  # (2700*6,10)
            loss = torch.sum(torch.max(pos - neg + self.eps, z))
        # loss = torch.sum(torch.max(pos - neg + self.eps, z))

        return loss,torch.sum(torch.max(pos - neg + self.eps, z)),torch.sum(torch.abs(torch.max(tmp, dim=1)[0]-1))



