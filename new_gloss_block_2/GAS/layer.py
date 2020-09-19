import torch
import torch.nn as nn

# lstm网络结构
class DynamicLSTM(nn.Module):
    ''' LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...). '''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias # True
        self.batch_first = batch_first # True
        self.dropout = dropout # 0
        self.bidirectional = bidirectional # False
        self.only_use_last_hidden_state = only_use_last_hidden_state # False
        self.rnn_type = rnn_type # LSTM
        
        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, x, x_len):
        """

        :param x: input_f_ebm (batchsize, 30, 300)
        :param x_len: 30 or 100
        :return:
        """
        '''
        sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort
        '''
        # （单词）序列的长度
        total_length = x.size(1) if self.batch_first else x.size(0) # the length of sequence
        ''' sort '''

        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        ''' pack '''
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        ''' process '''
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        ''' unsort '''
        ht = ht[:, x_unsort_idx]
        if self.only_use_last_hidden_state:
            return ht
        else:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first, total_length=total_length)
            if self.batch_first:
                out = out[x_unsort_idx]
            else:
                out = out[:, x_unsort_idx]
            if self.rnn_type == 'LSTM':
                ct = ct[:, x_unsort_idx]
            return out, (ht, ct)
    
