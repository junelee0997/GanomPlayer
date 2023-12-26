import torch
import torch.nn as nn
import math
class GE_BiL(nn.Module):
    def __init__(self, input_size, lstm_input, hidden_size,  device, ong_size = 1, stat_size = 2, health_size = 1, loc_size = 3, item_size = 3, num_layers = 1, relu=True):
        super(GE_BiL, self).__init__()
        self.input_size = input_size
        self.relu = relu
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ong_size = ong_size
        self.stat_size = stat_size
        self.health_size = health_size
        self.loc_size = loc_size
        self.item_size= item_size
        self.fc_health = nn.Linear(health_size * 4, input_size)
        self.fc_loc = nn.Linear(loc_size * 6, input_size)
        self.fc_stat = nn.Linear(stat_size * 2, input_size)
        self.fc_item = nn.Linear(item_size * 2, input_size)
        self.fc_ground = nn.Linear(ong_size * 2, input_size)
        self.fc_in = nn.Linear(input_size, lstm_input)
        self.lstm = nn.LSTM(lstm_input, hidden_size, num_layers, batch_first=True, bidirectional=True)
        '''self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)'''
    def foward(self, on_ground, stat, health_data, map_data, hand_item):
        deal = self.fc_health(health_data)
        map = self.fc_loc(map_data)
        act = self.fc_stat(stat)
        lev = self.fc_ground(on_ground)
        item = self.fc_item(hand_item)
        if self.relu:
            deal = torch.relu(deal)
            map = torch.relu(map)
            act = torch.relu(act)
            lev = torch.relu(lev)
            item = torch.relu(item)
        else:
            deal = torch.sigmoid(deal)
            map = torch.sigmoid(map)
            act = torch.sigmoid(act)
            lev = torch.sigmoid(lev)
            item = torch.sigmoid(item)
        put = torch.cat([deal, map, act, lev, item])
        put = self.fc_in(put)
        '''if self.relu: put = torch.relu(put)
        else: put = torch.sigmoid(put)'''
        h0 = torch.zeros(self.num_layers * 2, stat.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, stat.size(0), self.hidden_size).to(self.device)
        out = self.lstm(put, (h0, c0))
        return out

class GDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super(GDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.change = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)
    def foward(self, x):
        pass