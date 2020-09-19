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

class Run:
    def __init__(self, auxiliary_metrix, gloss_id, train_data,test_data, dict_data, config, word_to_id, glove_init_emb, backoff_len, backoff_right,o_len):
        # 相关配置
        self.config = config
        # 模型结构

        self.model = Modelmy(config, glove_init_emb, auxiliary_metrix, gloss_id).to("cuda")
        self.train_data = train_data
        self.dict_data = dict_data
        self.word_to_id = word_to_id
        self.test_data = test_data
        self.backoff_len = backoff_len # 815
        self.backoff_right = backoff_right # 734
        self.olen = o_len
        self.sense_mask = auxiliary_metrix[4].to("cuda")

        # 新增
        # self.old_train_data = old_train_data
        # self.auxiliary_metrix = auxiliary_metrix
        self.gloss_id = gloss_id

    def _print_params(self):
        n_embedding, n_lstm = 0, 0
        # named_parameters()给出网络层的名字和参数的迭代器
        for name, param in self.model.named_parameters():

            if param.requires_grad and "embedding" in name: #
           #  if param.requires_grad and "embedding" in name:
                # 计算embeddding的 参数个数
                n_embedding += torch.prod(torch.tensor(param.shape))
            elif param.requires_grad and "lstm" in name:
                # 计算lstm 层的参数个数
                n_lstm += torch.prod(torch.tensor(param.shape))

        print(f"n_embedding_params: {int(n_embedding)}, n_lstm_params: {int(n_lstm)}")
        n_trainable_params, n_nontrainable_params = 0, 0
        # 计算机所有的参数
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print(f"n_trainable_params: {int(n_trainable_params)}, n_nontrainable_params: {int(n_nontrainable_params)}")
        for name, param in self.model.named_parameters():

            if param.requires_grad:
                print("True",name,param.shape)
            else:
                print("False", name, param.shape)

    def train(self, optimizer, criterion):
        n_batch = 0
        n_correct = 0
        n_total = 0
        train_loss = 0  # same to global step
        train_loss1 = 0
        train_loss2 = 0
        # 切换状态
        self.model.train()

        for batch_id, batch_data in enumerate(
                batch_generator(True, self.config.batch_size, self.config.max_gloss_words, self.train_data, self.dict_data,
                                self.word_to_id['<pad>'], self.config.n_step_f, self.config.n_step_b,
                                pad_last_batch=True)):

            sentence, sense_ids, all_gloss, sense_mask, output_g, alpha= self.model(batch_id, batch_data, self.gloss_id, mode="train")
            # print("batch_id:{},alpha:{}".format(batch_id,alpha[0]))
            # 累计每个batch中的正确的个数
            # print(batch_id)
            n_correct += right_num(self.config, sentence, sense_ids, all_gloss, sense_mask, batch_id)
            n_total += batch_data[1].shape[0]
            optimizer.zero_grad()  # clear gradient accumulators
            # loss
            loss,loss1,loss2 = criterion(sentence, all_gloss, sense_mask, sense_ids, alpha, mode="train")
            train_loss += loss.cpu().item()
            train_loss1 += loss1.cpu().item()
            train_loss2 += loss2.cpu().item()
            n_batch += 1
            # 对alpha的梯度利用sense_mask进行置零，如果对比效果，可以注释
            # alpha.register_hook(_alpha_hook)

            loss.backward()  # compute gradients through back-propagation
            # print("alpha_grad:",alpha.grad)
            optimizer.step()
        return n_correct/n_total, train_loss/n_batch, output_g, train_loss1/n_batch,train_loss2/n_batch

    def _evaluate(self, criterion, output_g):
        self.model.eval()
        n_batch = 0
        n_correct = 0
        n_total = 0
        test_loss = 0
        test_loss1 = 0
        test_loss2 = 0

        with torch.no_grad():
            for batch_id, batch_data in enumerate(
                    batch_generator(False, self.config.batch_size, self.config.max_gloss_words, self.test_data,
                                    self.dict_data,self.word_to_id['<pad>'], self.config.n_step_f, self.config.n_step_b,
                                    pad_last_batch=True)):
                #  sentence, sense_ids, all_gloss, sense_mask = self.model(batch_data)

                sentence, sense_ids, all_gloss, sense_mask,alpha= self.model(batch_id, batch_data, output_g, mode="test")
                """已修改"""
                # loss = criterion(sentence, all_gloss, target_gloss, sense_mask, sense_ids)
                loss,loss1,loss2 = criterion(sentence, all_gloss, sense_mask, sense_ids, alpha, mode="test")
                test_loss += loss.cpu().item()
                test_loss1 += loss1.cpu().item()
                test_loss2 += loss2.cpu().item()

                n_correct += right_num(self.config, sentence, sense_ids, all_gloss, sense_mask, batch_id)
                n_total += batch_data[1].shape[0]
                n_batch += 1

        p = (n_correct + self.backoff_right) / (n_total + self.backoff_len)
        r = (n_correct + self.backoff_right) / self.olen
        f1 = 2 * p * r / (p + r)
        return n_correct/n_total, f1, test_loss/n_batch,test_loss1/n_batch,test_loss2/n_batch


    def start(self):
        # 筛选requeired_grad 为true的参数
        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        # 优化器
        # optimizer = torch.optim.Adam(_params, lr=self.config.lr_start, weight_decay=self.config.lambda_l2_reg)
        optimizer = torch.optim.Adam(_params, lr=self.config.lr_start)
        # 定义损失函数
        margin = 0.75
        criterion = Marginal_loss(margin, self.config)
        print("margin loss:",margin)
        # 打印的参数信息
        self._print_params()
        # 训练次数
        for epoch in range(1, self.config.n_epochs + 1):
            print('::: EPOCH: %d :::' % epoch)
            # 训练数据
            print("trianing...")
            train_acc, train_loss, output_g ,train_loss1,train_loss2= self.train(optimizer, criterion)
            # 测试数据if (epoch == 50):
            print("testing...")
            test_acc, test_f1, test_loss,test_loss1,test_loss2 = self._evaluate(criterion, output_g)
            # 当第20轮时，保存到文件
            # print([x.grad for x in optimizer.param_groups[0]['params']])
            print(f"train_acc:{train_acc}, train_loss:{train_loss}")
            print(f"train_loss1:{train_loss1}, train_loss2:{train_loss2}")
            print(f"test_acc:{test_acc}, test_f1:{test_f1}, test_loss:{test_loss}")
            print(f"test_loss1:{test_loss1}, test_loss2:{test_loss2}")



if __name__ == "__main__":

    print("sys.argv",sys.argv)


    if len(sys.argv) > 1:
        config.random_config()  # random search to find best parameters
    # vars(config) ： object.filed -> dict
    print (vars(config))

    # Load data
    # ==================================================================================================================

    print('Loading all-words task data...')
    train_dataset = _path.ALL_WORDS_TRAIN_DATASET[0]  # semcor
    print ('train_dataset: ' + train_dataset)
    val_dataset = _path.ALL_WORDS_VAL_DATASET  # semeval2007
    print ('val_dataset: ' + val_dataset)
    test_dataset = _path.ALL_WORDS_TEST_DATASET[1]  # senseval2
    print ('test_dataset: ' + test_dataset)

    train_data = load_train_data(train_dataset)
    val_data = load_val_data(val_dataset)
    test_data = load_test_data(test_dataset)
    o_len = len(test_data)
    print ('O Original dataset size (train/val/test): %d / %d / %d' % (len(train_data), len(val_data), len(test_data)))

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
        print ('1 Filtered dataset size (train/val/test): %d / %d / %d' % (
            len(train_data), len(val_data), len(test_data)))
        # back_off_result 存储与new_test_data中target word重合的部分
        print ('***Test using back-off instance: %d' % (len(back_off_result)))
        # 丢弃的test 数量
        missed = test_data_lenth_pre - (len(test_data) + len(back_off_result))
        # 丢失率
        missed_ratio = float(missed) / test_data_lenth_pre
        #
        print ('***Test missed instance(not in MFS/FS): %d/%d = %.3f' % (
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
    print ('Avg n senses per target word: %.4f' % average_sense)
    with open('../tmp/target_word.txt', 'w') as f:
        for word, id in target_word_to_id.items():
            f.write('{}\t{}\n'.format(word, id))

    train_data = convert_to_numeric(train_data, word_to_id, target_word_to_id, target_sense_to_id, mode='Train')
    val_data = convert_to_numeric(val_data, word_to_id, target_word_to_id, target_sense_to_id, mode='Val')
    test_data = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, mode='Test')
    print ('2 After convert_to_numeric dataset size (train/val/test): %d / %d / %d' % (
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
    # print(gloss_id)
    # print(backoff_len, backoff_right, o_len)
    # min_gloss = min_gloss_every_word(auxiliary_metrix)

    if config.use_pre_trained_embedding:
        print('Load pre-trained word embedding...')
        # 所有单词的embedding
        glove_init_emb = fill_with_gloves(word_to_id,  _path.GLOVE_VECTOR, emb_size=config.embedding_size,vocab_size='42B')
    else:
        glove_init_emb = None


    run = Run(auxiliary_metrix,  gloss_id, train_data, test_data, dict_data, config, word_to_id, glove_init_emb, backoff_len, backoff_right, o_len)
    run.start()


