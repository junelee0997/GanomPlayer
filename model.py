import torch
import torch.nn as nn
import math

class EncGen(nn.Module):
    def __init__(self, dropout, device, input_size, lstm_input, hidden_size, stat_size=4, vel_size=4, oppo_size=6, num_layer = 1): #
        super().__init__()
        self.dropout = dropout
        self.device = device
        self.stat_size = stat_size
        self.vel_size = vel_size
        self.oppo_size = oppo_size
        self.input_size = input_size
        self.lstm_input = lstm_input
        self.hidden_size = hidden_size
        self.attentLinVel = nn.Linear(vel_size, stat_size, device=device)
        self.attentLinOppo = nn.Linear(oppo_size, stat_size, device=device)
        self.attentSoft = nn.Softmax()
        self.oppoGrid = nn.Linear(oppo_size, input_size, device=device)
        self.oppoMid = nn.Linear(2 * oppo_size, input_size, device=device)
        self.velstatCNN = nn.Conv2d(1, input_size, 2, stride=1, device=device)
        self.outCNN = nn.Conv2d(1, input_size, 4, stride = 1, device = device)
        self.lstm_fn = nn.Linear(input_size, lstm_input, device = device)
        self.lstm = nn.LSTM(lstm_input, hidden_size, dropout=dropout, batch_first=True, num_layers=num_layer, device=device)

        self.velstatRelu = nn.ReLU()
        self.gridRelu = nn.ReLU()
        self.midRelu = nn.ReLU()
        self.outRelu = nn.ReLU()
    def forward(self, stat, velocity, opponentGrid, opponentMid): # stat [onhit ... ], velocity [pith, vel], opponentgrid[Area], opponentMid[coord]
        attention = self.attentLinVel(velocity) + self.attentLinOppo(opponentGrid)
        attention = self.attentSoft(attention)
        attention_stat = stat.view([-1]) * attention.view([-1])
        attention_stat = attention_stat.view([1, -1, self.stat_size])
        Feature1 = self.velstatCNN(torch.stack((attention_stat, velocity)))
        Feature1 = self.velstatRelu(Feature1)
        Feature1 = torch.squeeze(Feature1)
        Feature1 = torch.sum(Feature1, dim=0)
        Feature2 = self.oppoGrid(opponentGrid)
        Feature2 = self.gridRelu(Feature2)
        Feature3 = self.oppoMid(opponentMid.view(-1, self.oppo_size * 2).type(torch.DoubleTensor))
        Feature3 = self.midRelu(Feature3)
        Feature = torch.concat((torch.stack([Feature1], Feature2), Feature3))
        x = self.outCNN(Feature)
        x = self.outRelu(x)
        x = torch.squeeze(x)
        x = torch.sum(x, dim=0)
        x = self.lstm_fn(x)
        output, _, _ = self.lstm(x)
        return output
class ManyToOne(nn.Module): # 
    def __init__(self, encoder, output, hidden_size, seq_len = 10):
        super().__init__()
        self.enc = encoder
        self.rot_fn1 = nn.Linear(output * seq_len, hidden_size, device=encoder.device)
        self.rot_sig = nn.Sigmoid()
        self.rot_fn2 = nn.Linear(hidden_size, 2, device=encoder.device)
        self.vel_fn1 = nn.Linear(output * seq_len, hidden_size, device=encoder.device)
        self.vel_sig = nn.Sigmoid()
        self.vel_fn2 = nn.Linear(hidden_size, 3, device=encoder.device)
        self.snk_fn1 = nn.Linear(output * seq_len, hidden_size, device=encoder.device)
        self.snk_fn2 = nn.Linear(hidden_size, 1, device=encoder.device)
        self.snk = nn.Sigmoid()
        self.spt_fn1 = nn.Linear(output * seq_len, hidden_size, device=encoder.device)
        self.spt_fn2 = nn.Linear(hidden_size, 1, device=encoder.device)
        self.spt = nn.Sigmoid()
        self.atk_fn1 = nn.Linear(output * seq_len, hidden_size, device=encoder.device)
        self.atk_fn2 = nn.Linear(hidden_size, 1, device=encoder.device)
        self.atk = nn.Sigmoid()

    def forward(self, stat, vel, opponentGrid, opponentMid):
        output = self.enc(stat, vel, opponentGrid, opponentMid)
        output = output.reshape(output.shape[0], -1)
        rotation = self.rot_fn2(self.rot_sig(self.rot_fn1(output)))
        velocity = self.vel_fn2(self.vel_sig(self.vel_fn1(output)))
        isSneaking = self.snk(self.snk_fn2(self.snk_fn1(output)))
        isSprinting = self.spt(self.spt_fn2(self.spt_fn1(output)))
        attackIndex = self.atk(self.atk_fn2(self.atk_fn1(output)))
        return rotation, velocity, isSneaking, isSprinting, attackIndex

class discriminator(nn.Module):
    def __init__(self, dropout, device, input_size, lstm_input, hidden_size, out_hidden, stat_size=4, vel_size=4, num_layer = 1, seq_len=10):
        super().__init__()
        self.dropout = dropout
        self.device = device
        self.stat_size = stat_size
        self.vel_size = vel_size
        self.input_size = input_size
        self.lstm_input = lstm_input
        self.hidden_size = hidden_size
        self.attentLinVel = nn.Linear(vel_size, stat_size, device=device)
        self.attentSoft = nn.Softmax()
        self.velstatCNN = nn.Conv2d(1, input_size, 2, stride=1, device=device)
        self.outCNN = nn.Conv2d(1, input_size, 4, stride=1, device=device)
        self.lstm_fn = nn.Linear(input_size, lstm_input, device=device)
        self.lstm = nn.LSTM(lstm_input, hidden_size, dropout=dropout, batch_first=True, num_layers=num_layer, device=device)
        self.velstatRelu = nn.ReLU()
        self.outRelu = nn.ReLU()

        self.out_fn = nn.Linear(hidden_size * seq_len, out_hidden, device=device)
        self.out_fn2 = nn.Linear(out_hidden, 1, device=device)
        self.outSig = nn.Sigmoid()
    def forward(self, stat, velocity, opponentGrid, opponentMid):
        attention = self.attentlLinVel(velocity)
        attention = self.attentSoft(attention)
        attention_stat = torch.matmul(stat, attention)
        Feature1 = self.velstatCNN(torch.stack((attention_stat, velocity)))
        Feature1 = self.velstatRelu(Feature1)
        Feature1 = torch.squeeze(Feature1)
        Feature = torch.sum(Feature1, dim=0)
        x = self.outCNN(Feature)
        x = self.outRelu(x)
        x = torch.squeeze(x)
        x = torch.sum(x, dim=0)
        x = self.lstm_fn(x)
        output, _, _ = self.lstm(x)
        output = output.reshape(output.shape[0], -1)
        output = self.outSig(self.out_fn2(self.out_fn(output)))
        return output


'''class DecGEN(nn.Module):
    def __init__(self, hidden_size, output_size, lstm_input, device):
        super().__init__()
        self.lstm = nn.LSTM(lstm_input, hidden_size, batch_first=True, device=device)
        self.fc = nn.Linear(hidden_size, output_size)
    def foward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        return output, hidden, cell
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output, hidden_size, device):
        super().__init__()
        self.enc = encoder
        self.dec = decoder
        self.rot_fn1 = nn.Linear(output, hidden_size, device=device)
        self.rot_fn2 = nn.Linear(hidden_size, 2, device = device)
        self.vel_fn1 = nn.Linear(output, hidden_size, device=device)
        self.vel_fn2 = nn.Linear(hidden_size, 3, device=device)
        self.snk_fn1 = nn.Linear(output, hidden_size, device=device)
        self.snk_fn2 = nn.Linear(hidden_size, 1, device=device)
        self.snk = nn.Sigmoid()
        self.spt_fn1 = nn.Linear(output, hidden_size, device=device)
        self.spt_fn2 = nn.Linear(hidden_size, 1, device=device) 
        self.spt = nn.Sigmoid()
        self.atk_fn1 = nn.Linear(output, hidden_size, device=device)
        self.atk_fn2 = nn.Linear(hidden_size, 1, device=device)
        self.atk = nn.Sigmoid()
    def forward(self,  stat, vel, opponentGrid, opponentMid):
        output, hidden, cell = self.enc(stat, vel, opponentGrid, opponentMid)
        for()
        output, _, _ = self.dec(output, hidden, cell)
        rotation = self.rot_fn2(self.rot_fn1(output))
        velocity = self.vel_fn2(self.vel_fn1(output))
        isSneaking = self.snk(self.snk_fn2(self.snk_fn1(output)))
        isSprinting = self.spt(self.spt_fn2(self.spt_fn1(output)))
        attackIndex = self.atk(self.atk_fn2(self.atk_fn1(output)))
        return rotation, velocity, isSneaking, isSprinting, attackIndex'''

'''class EncGenBi(nn.Module): #Encoder of Generator using BiLSTM
    def __init__(self, dropout, input_size, lstm_input, hidden_size,  device, ong_size = 1, stat_size = 2, health_size = 1, loc_size = 3, item_size = 3, num_layers = 1, relu=True):
        super().__init__()
        self.dropout = dropout
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
        self.fc_health = nn.Linear(health_size * 4, input_size, device=device)
        self.fc_loc = nn.Linear(loc_size * 6, input_size, device=device)
        self.fc_stat = nn.Linear(stat_size * 2, input_size, device=device)
        self.fc_item = nn.Linear(item_size * 2, input_size, device=device)
        self.fc_ground = nn.Linear(ong_size * 2, input_size, device=device)
        self.fc_in = nn.Linear(input_size, lstm_input, device=device)
        self.lstm = nn.LSTM(lstm_input, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True, device=device)'''
'''
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)'''
'''
    def foward(self, on_ground, stat, health_data, map_data, hand_item):
        deal = self.fc_health(health_data)
        map = self.fc_loc(map_data)
        act = self.fc_stat(stat)
        lev = self.fc_ground(on_ground)
        item = self.fc_item(hand_item)
        if self.relu:
            deal = torch.relu(deal).to(self.device)
            map = torch.relu(map).to(self.device)
            act = torch.relu(act).to(self.device)
            lev = torch.relu(lev).to(self.device)
            item = torch.relu(item).to(self.device)
        else:
            deal = torch.sigmoid(deal).to(self.device)
            map = torch.sigmoid(map).to(self.device)
            act = torch.sigmoid(act).to(self.device)
            lev = torch.sigmoid(lev).to(self.device)
            item = torch.sigmoid(item).to(self.device)
        put = torch.cat([deal, map, act, lev, item]).to(self.device)
        put = self.fc_in(put)'''
'''
        if self.relu: put = torch.relu(put)
        else: put = torch.sigmoid(put)'''
'''
        h0 = torch.zeros(self.num_layers * 2, stat.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, stat.size(0), self.hidden_size).to(self.device)
        out = self.lstm(put, (h0, c0))
        return out

class DecGen(nn.Module): #Decoder of Generator
    def __init__(self, output_size, hidden_size, num_layers, dropout, device):
        super().__init__()
        self.device = device
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.change = nn.Linear(hidden_size, hidden_size, device=device)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True, dropout=dropout, device=device)
        self.fc_out = nn.Linear(hidden_size, output_size, device=device)
    def foward(self, x):
        pass

class GenSeq2Seq(nn.Module): # Seq2Seq of Generator
    pass

class EncDisc(nn.Module): # Encoder of Discriminator
    pass
class DecDisc(nn.Module): # Decoder of Discriminator
    pass'''

'''
    isOnDamage (bool)
    isOnGround (bool)
    isSneaking (bool)
    isSprinting (bool) 
    pitch (float)
    velocity (float * 3)
    
    image (array) by pyautogui => 
    
    output
    rotation (float * 2)
    velocity (float * 3)
    isSneaking (bool)
    isSprinting (bool)
    attackIndex (int) 0 or -1
'''