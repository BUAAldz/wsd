import torch
import torch.nn
import torch.nn.functional as F


def right_num (config, sentence, sense_ids, all_gloss, sense_mask, batch_id):
    """
    :param config:
    :param sentence: （8,300）
    :param sense_ids:（8，）
    :param all_gloss:（8，10,300）
    :param sense_mask: （8，10）
    :return:
    """
    # 计算距离，用10填充
    # (8,10)，和每个词义的距离
    batch_size = sentence.shape[0]
    distence = torch.ones((batch_size, config.max_n_sense), device="cuda")
    for i in range(batch_size):
        # 计算机距离 sentence(8,300)-》sentence[i]:(300)-》(1,300)-》（10,300） 和 gloss_sense[i] (10,300)
        # distence[i]  每个data中每个词义的距离 每次计算10个词意的距离
        distence[i] = F.pairwise_distance(sentence[i].reshape(1, -1).expand(config.max_n_sense, -1),all_gloss[i])  # (max_n_snese)看一下
    # （8,10）
    big = torch.ones((batch_size, config.max_n_sense), device='cuda') * 10 #用10填充

    # result shape (8,10) 把默认的sense的地方填充10（越大越好）
    result = torch.where(sense_mask != 0, distence, big)  # (batchsize, max_n_sense)
    # 选择距离最小的作为result_id
    result_ids = torch.argmin(result, dim=1)
    try:
        right = (sense_ids.cpu() == result_ids.cpu()).sum().item()
    except Exception as e:
        print("ERROR！！！:",e)
        right = sense_ids.cpu().sum().item()

    # 正确的个数
    # if batch_id == 2:
    #     print("sentence")
    #     print(sentence[500])
    #     print("all_gloss")
    #     print(all_gloss[500])
    #     print("result")
    #     print(result[500])
    #     print("result_ids")
    #     print(result_ids[500])
    #     print("sense_ids")
    #     print(sense_ids[500])

    return right

def min_gloss_every_word(auxiliary_metrix):
    sense_to_gloss_id = auxiliary_metrix[0].to("cuda")  # (2979)
    word_to_sense_id = auxiliary_metrix[1].to("cuda")  # (654,10)
    gloss_to_word_id = auxiliary_metrix[2].to("cuda")  # (2700,6)

    count = []
    for i in range(word_to_sense_id.shape[0]):
        a = torch.gather(torch.cat((torch.zeros((1),dtype=torch.int64).to("cuda"),sense_to_gloss_id), dim=0), 0, word_to_sense_id[i]) #(10)
        b = torch.gather(torch.cat((torch.zeros((1,6),dtype=torch.int64).to("cuda"),gloss_to_word_id),dim=0), 0, a.unsqueeze(-1).expand(-1,6)) #(10,6)
        c = torch.gather(torch.cat((torch.zeros((1,10),dtype=torch.int64).to("cuda"),word_to_sense_id),dim=0), 0, b.reshape(60,-1).expand(-1,10)) #(60,10)
        d = torch.gather(torch.cat((torch.zeros((1),dtype=torch.int64).to("cuda"),sense_to_gloss_id), dim=0), 0, c.reshape(600)) #(60,10)
        m = torch.where(d!=0, torch.ones_like(d).to("cuda"), d)
        count.append(torch.sum(m).item())
    print(count)
    print(max(count))
    print(min(count))
    return count


def right_num_test (config, sentence, sense_ids, all_gloss, sense_mask, ids, old_test_data, old_gloss_dict,old_word_to_sense,badcase_list):
    """
    :param config:
    :param sentence: （8,300）
    :param sense_ids:（8，）
    :param all_gloss:（8，10,300）
    :param sense_mask: （8，10，1024）
    :return:
    """
    # 计算距离，用10填充
    # (8,10)，和每个词义的距离
    distence = torch.ones((config.batch_size, config.max_n_sense), device="cuda")
    for i in range(config.batch_size):
        # 计算机距离 sentence(8,300)-》sentence[i]:(300)-》(1,300)-》（10,300） 和 gloss_sense[i] (10,300)
        # distence[i]  每个data中每个词义的距离 每次计算10个词意的距离
        distence[i] = F.pairwise_distance(sentence[i].reshape(1, -1).expand(config.max_n_sense, -1),all_gloss[i])  # (max_n_snese)看一下
    # （8,10）
    big = torch.ones((config.batch_size, config.max_n_sense), device='cuda') * 10 #用10填充
    sense_mask = torch.sum(sense_mask, dim=2)  # (batchsize, max_n_sense)
    # result shape (8,10) 把默认的sense的地方填充10（越大越好）
    result = torch.where(sense_mask != 0, distence, big)  # (batchsize, max_n_sense)
    # 选择距离最小的作为result_id
    # (8,)
    result_ids = torch.argmin(result, dim=1)
    try:


        # 收集badcase
        for i in range(sense_ids.size(0)):
            if sense_ids[i].item() != result_ids[i].item():
                badcase_id = ids[i]
                elem=old_test_data[badcase_id]
                # list -> str
                elem['context_str'] = ' '.join(elem['context'])
                # 添加真实词义
                elem["true_gloss"] = old_gloss_dict[elem['target_sense']]
                # 词义的index
                elem["true_gloss_id"] = sense_ids[i].item()
                # 添加错误词义
                false_gloss_name=old_word_to_sense[elem['target_word']][result_ids[i].item()]
                elem['false_gloss'] = old_gloss_dict[false_gloss_name]
                # 词义的index
                elem['false_gloss_id'] = result_ids[i].item()
                # 词义数量
                elem['sense_num'] = len(old_word_to_sense[elem['target_word']])
                badcase_list.append(elem)

        right = (sense_ids.cpu() == result_ids.cpu()).sum().item()
    except Exception as e:
        print("ERROR！！！:",e)
        right = sense_ids.cpu().sum().item()
    # right = (sense_ids.cpu() == result_ids.cpu()).sum().item()
    # 正确的个数
    return right



