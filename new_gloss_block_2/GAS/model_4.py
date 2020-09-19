import numpy as np
import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
from GAS.layer import DynamicLSTM
import torch.nn.functional as F

# 定义网络结构
class Modelmy(nn.Module):
    def __init__(self, config, glove_inti_embedding, auxiliary_metrix, gloss_id):
        super(Modelmy, self).__init__()
        '''config'''
        # self.batch_size = config.batch_size
        self.n_step_f = config.n_step_f
        self.n_step_b = config.n_step_b
        self.embedding_size = config.embedding_size
        self.max_n_sense = config.max_n_sense
        self.max_gloss_words = config.max_gloss_words
        self.drop_out = config.dropout
        self.HD = config.LSTMHD  # emp_shape # 300=>100
        self.sense_to_gloss_id = auxiliary_metrix[0].to("cuda")  # (2979)
        self.word_to_sense_id = auxiliary_metrix[1].to("cuda")  # (653,10)
        self.gloss_to_word_id = auxiliary_metrix[2].to("cuda")  # (2700,6)
        self.gloss_to_word_mask = auxiliary_metrix[3].to("cuda")  # (2700,100)
        self.sense_mask = auxiliary_metrix[4].to("cuda")  # (2700,6,10)
        self.hook_mask = self.sense_mask
        self.init_alpha= auxiliary_metrix[5].to("cuda")  # (2700,6,10)
        self.gloss_id = gloss_id

        '''embedding'''
        # []
        #  First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
        self.glove_embedding = nn.Embedding.from_pretrained(glove_inti_embedding)
        self.pos_embedding = nn.Embedding(16, 8, padding_idx=0)
        """dropout"""
        self.Dropout = nn.Dropout(self.drop_out)

        '''layer'''
        # HD 代表h设定的h的维度，并且每一个时间节点t，都会有一个输出，一般情况采用最后时刻的输出，当然也可以利用各个时间点的hidden层的特征
        self.lstm0 = DynamicLSTM(self.embedding_size+8, self.HD, num_layers=1)
        self.lstm1 = DynamicLSTM(self.embedding_size+8, self.HD, num_layers=1)
        self.lstm2 = DynamicLSTM(self.embedding_size, self.HD, num_layers=1)

        # """alpha"""
        # (2700,6,10)
        self.alpha = torch.where(self.sense_mask==0, torch.zeros_like(self.sense_mask,requires_grad=False),self.init_alpha)
        # self.alpha.requires_grad=True
        self.alpha = nn.Parameter(self.alpha,requires_grad=True)
        self.mask = torch.sum(self.alpha, dim=-1).reshape(2700,6,-1).expand(-1,-1,self.max_n_sense)
        self.zeros_alpha = torch.zeros_like(self.alpha, requires_grad=False)
        self.ones_alpha = torch.ones_like(self.alpha, requires_grad=False)



    def _alpha_hook(self, grad):
        grad = torch.where(self.hook_mask == 0, self.hook_mask, grad)
        return grad

    def forward(self, batch_id, batchdata, test_gloss_list, mode):

        """

        :param batchdata:
            (forward_text, backward_text, xfbs, sense_ids, glosses_ids, glosses_lenth, sense_mask)
        :return:
        """
        inputs_f = batchdata[0].to("cuda")  # (batchsize, n_step_f)
        inputs_b = batchdata[1].to("cuda")  # (batchsize, n_step_b)
        sense_ids = batchdata[2].to("cuda")  #(batchsize)目标词义的位置，第几个词义
        glosses = batchdata[3].to("cuda") # (batchsize, max_n_sense, max_gloss_words)
        sense_masks = batchdata[4].to("cuda")  # (batchsize, max_n_sense)
        pos_f = batchdata[5].to("cuda")
        pos_b = batchdata[6].to("cuda")
        batch_size = inputs_f.shape[0]

        """长度计算"""
        inputs_f_len = torch.sum(inputs_f != 0, dim=-1)  # (batchsize)
        inputs_b_len = torch.sum(inputs_b != 0, dim=-1)  # (batchsize)
        ones = torch.ones_like(inputs_f_len)
        inputs_f_len = torch.max(inputs_f_len,ones)
        inputs_b_len = torch.max(inputs_b_len,ones)

        """原始句子压缩"""
        # (8,30,300)
        inputs_f_emb = self.glove_embedding(inputs_f)# (batchsize, n_step_f , 300)
        inputs_f_emb = self.Dropout(inputs_f_emb)
        inputs_b_emb = self.glove_embedding(inputs_b)# (batchsize, n_step_b, 300)
        inputs_b_emb = self.Dropout(inputs_b_emb)

        pos_f_emb = self.pos_embedding(pos_f)  # (batchsize, n_step_f ,8)
        pos_b_emb = self.pos_embedding(pos_b)  # (batchsize, n_step_b, 8)

        f_emb = torch.cat((inputs_f_emb, pos_f_emb), dim=-1)  # (batchsize, n_step_f ,308)
        b_emb = torch.cat((inputs_b_emb, pos_b_emb), dim=-1)  # (batchsize, n_step_b ,308)

        _, (forword_t, _) = self.lstm0(f_emb, inputs_f_len)  # (batchsize, 30, 300)
        _,(back_t, _) = self.lstm1(b_emb, inputs_b_len)#(batchsize, 30, 300)

        sentence = torch.max(forword_t.squeeze(0) ,back_t.squeeze(0)).to("cuda")


        """gloss压缩"""
        if mode == "train":   #训练模式
            #gloss_id (2700,100)
            all_gloss_len = torch.sum(self.gloss_id != 0, dim=-1) #(2700)
            input_g= self.glove_embedding(self.gloss_id)  #(2700, max_gloss_word, embeddingsize)
            gloss_to_word_mask = self.gloss_to_word_mask.reshape(2700, 100, 1).expand(-1, -1, 300)

            for num in range(3):  #2层block模块
                #得到gloss表征
                _,(g, _) = self.lstm2(input_g, all_gloss_len)  # (2700, HDsize)
                g = torch.cat((torch.zeros(1,self.HD).to("cuda"), g.reshape(2700,-1)), dim=0) # (2701, HDsize)
                #得到sense表征
                a = torch.gather(g.reshape(2701,-1), 0, self.sense_to_gloss_id.reshape(-1, 1).expand(-1, 300))  # (2979, 300)
                a = torch.cat((torch.zeros(1,self.HD).to("cuda"), a), dim=0)# (2980, 300)
                #得到target_word表征
                b = torch.gather(a.reshape(-1,1,300).expand(-1,self.max_n_sense,-1), 0, self.word_to_sense_id.reshape(654, -1, 1).expand(-1, -1, 300))  # (654, 10, 300)
                b = torch.cat((torch.zeros(1,self.max_n_sense, self.HD).to("cuda"), b), dim=0)# (655, 10, 300)
                #得到指定gloss中的target_word表征
                c = torch.gather(b.reshape(655,1,self.max_n_sense,-1).expand(-1,6,-1,-1), 0, self.gloss_to_word_id.reshape(2700, -1, 1, 1).expand(-1, -1,self.max_n_sense, 300))  # (2700, 6, 10, 300)
                #加权求和

                temp = torch.where(self.mask == 0, self.ones_alpha, self.alpha)
                #####
                self.alpha1 = torch.where(self.mask == 0, self.zeros_alpha , temp/torch.sum(temp, dim=-1).reshape(2700,6,1).expand(-1,-1,self.max_n_sense))  #线性归一化

                d = torch.sum(c * (self.alpha1.reshape(2700,6,-1,1).expand(-1, -1, -1, 300)), dim=2)  # (2700, 6, 300)
                d = torch.cat((torch.zeros(2700,1,self.HD).to("cuda"), d),dim=1) # (2700, 7, 300)
               #替换gloss中的targetword表征
                f = torch.gather(d, 1, gloss_to_word_mask)# (2700, 100, 300)
                input_g = torch.where(gloss_to_word_mask==0, input_g, f)# (2700, 100, 300)

            # self.alpha2 = nn.functional.softmax(self.alpha1, dim = -1)
            #####
            self.alpha2 = torch.where(self.mask == 0, self.zeros_alpha , self.alpha1/torch.sum(self.alpha1, dim=-1).reshape(2700,6,1).expand(-1,-1,self.max_n_sense))   # 线性归一化
            # print("alpha")
            # print(self.alpha)
            # print("alpha1")
            # print(self.alpha1)
            # print("alpha2")
            # print(self.alpha2)

            #生成所有2700个gloss的表征
            _,(output_g,_) = self.lstm2(input_g, all_gloss_len)# (2700, HDsize)

            #在表中索引输入数据的gloss
            glosses = glosses.reshape(batch_size * self.max_n_sense, self.max_gloss_words)  # (80,100)
            index = torch.zeros(batch_size * self.max_n_sense, dtype=torch.int64).to("cuda")  # (80)
            for i in range(2700):
                tmp = self.gloss_id[i].expand(batch_size * self.max_n_sense, -1) #(5120,100)
                k = torch.ones(batch_size * self.max_n_sense, dtype=torch.int64).to("cuda") * (i+1)
                index = torch.where(torch.sum(glosses==tmp,dim=-1) == self.max_gloss_words, k, index)

            all_gloss = torch.gather(torch.cat((torch.zeros(1,300).to("cuda"),output_g.reshape(2700,-1)),dim=0), 0, index.reshape(-1,1).expand(-1,300)) #(5120, 300)
            all_gloss = all_gloss.reshape(batch_size,self.max_n_sense,-1) #(batchsize, max_n_sense, HD)

            # oo = torch.zeros_like(self.hook_mask)
            # self.hook_mask = torch.where(self.alpha1 > 0, self.hook_mask, oo)
            # self.alpha.register_hook(self._alpha_hook)
            # self.alpha1.register_hook(self._alpha_hook)
            # self.alpha2.register_hook(self._alpha_hook)

            return sentence, sense_ids, all_gloss, sense_masks, output_g, self.alpha2

        else:   #测试模式
            glosses = glosses.reshape(batch_size * self.max_n_sense, self.max_gloss_words)  # (80,100)
            index = torch.zeros(batch_size * self.max_n_sense, dtype=torch.int64).to("cuda")  # (5120)
            for i in range(2700):
                tmp = self.gloss_id[i].expand(batch_size * self.max_n_sense, -1)  # (5120,100)
                k = torch.ones(batch_size * self.max_n_sense, dtype=torch.int64).to("cuda") * (i + 1)
                index = torch.where(torch.sum(glosses == tmp, dim=-1) == self.max_gloss_words, k, index)

            all_gloss = torch.gather(torch.cat((torch.zeros(1, 300).to("cuda"), test_gloss_list.reshape(2700, -1)), dim=0), 0,
                                     index.reshape(-1, 1).expand(-1, 300))  # (5120, 300)
            all_gloss = all_gloss.reshape(batch_size,self.max_n_sense, -1)

            #打印alpha
            # for i in range(20):
            #     np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            #     if torch.sum(self.alpha2[i][1])!= 0:
            #         print(self.alpha2[i][1][:].cpu().numpy())

            return sentence, sense_ids, all_gloss, sense_masks, self.alpha2




