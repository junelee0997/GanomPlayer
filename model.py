import torch
import torch.nn as nn
import math
import numpy as np

class Generator(nn.Module):
    def __init__(self, device, output_size=12, input_size=7):
        super().__init__()
        self.device = device
        self.image_layer = nn.Sequential(
            # [100,3,140,140] -> [100,8,136,136]
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, device=device),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [100,8,68,68] -> [100,16,64,64]
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, device=device),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [100,16,32,32] -> [100,32,28,28]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, device=device),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [100,32,28,28] -> [100,32,14,14]
        )
        # [100,32 * 14 * 14] -> [100, 64]
        self.fc1_image = nn.Linear(32 * 14 * 14, 64, device=device)

        # [100, 7] -> [100, 32]
        self.fc1 = nn.Linear(input_size, 32, device=device)

        # after concatenate fc1_image, fc1, noise(100, 32)
        # [100, 128] -> [100, 64]
        self.fc2 = nn.Linear(128, 64, device=device)

        # [100, 64] -> [100, 64]
        self.fc3 = nn.Linear(64, 64, device=device)

        # [100, 64] -> [100, 9]
        self.out_fc = nn.Linear(64, output_size, device=device)
        self.softWS = nn.Softmax()
        self.softAD = nn.Softmax()
        self.space = nn.Sigmoid()
        self.shift = nn.Sigmoid()
        self.ctrl = nn.Sigmoid()
        self.atk = nn.Sigmoid()
    def forward(self, img, move1, move2, jmp, run, crh, del_yaw, del_pitch): # data shape : [1, ~]
        data = self.image_layer(img)
        data = torch.reshape(data, (-1, 32 * 14 * 14))
        data = self.fc1_image(data)

        action = torch.cat([move1, move2, jmp, run, crh, del_yaw, del_pitch], dim=1)
        data2 = self.fc1(action)

        z = torch.randn(1, 32, device=self.device)

        x = torch.cat([data, data2, z], dim = 1)

        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out_fc(x)

        move1 = self.softWS(x[:, :3])
        move2 = self.softAD(x[:, 3:6])
        jmp = self.space(x[:, 6]).unsqueeze(dim = 0).detach()
        run = self.ctrl(x[:, 7]).unsqueeze(dim = 0).detach()
        crh = self.shift(x[:, 8]).unsqueeze(dim = 0).detach()
        del_yaw = 20 * torch.tanh(x[:, 9]).unsqueeze(dim = 0).detach() # change of yaw(-10~10)
        del_pitch = 10 * torch.tanh(x[:, 10]).unsqueeze(dim = 0).detach() # change of pitch(-10~10)
        atkind = self.atk(x[:, 11]).unsqueeze(dim = 0).detach() # attack or not

        return move1, move2, jmp, run, crh, del_yaw, del_pitch, atkind
class Discriminator(nn.Module):
    def __init__(self, device, input_size=12, output_size=1):
        super().__init__()
        self.device = device
        self.image_layer = nn.Sequential(
            # [1, 10,3,140,140] -> [1, 10,8,136,136]
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, device=device),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [1, 10,8,68,68] -> [1, 10,16,64,64]
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, device=device),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [10,16,32,32] -> [10,32,28,28]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, device=device),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [10,32,28,28] -> [10,32,14,14]
        )
        # [1, 10, 12] -> [1, 10, 64]
        self.fc1 = nn.Linear(input_size, 64, device=device)

        #[1, 10, 32 * 14 * 14 + 64] -> [1, 10, 256]
        self.input_layer = nn.Linear(32 * 14 * 14 + 64, 128, device=device)
        self.lstm = nn.LSTM(128, 512, batch_first=True, device=device)

        #[1, 10240] -> [1, 512] -> 1
        self.fc2 = nn.Linear(5120, 128, device=device)
        self.fc3 = nn.Linear(128, 1, device=device)
        self.outsig = nn.Sigmoid()

    def forward(self, img, move1, move2, jmp, run, crh, del_yaw, del_pitch, atkind): # data shape : [1, 10, ~]
        data = self.image_layer(img)
        data = torch.reshape(data, (1, 10, -1))

        act = torch.cat([move1, move2, jmp * 3, run * 3, crh * 3, del_yaw * 3, del_pitch, atkind], dim=2)
        act = self.fc1(act)

        x = torch.cat([data, act], dim=2)
        x = self.input_layer(x)
        x = self.lstm(x)[0]
        x = torch.reshape(x, (1, -1))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.outsig(x)
        out = x.squeeze(dim=0).detach()
        return out

'''
    input
    WSmove: W or S or stop [float * 3](one-hot encoding)
    ADmove: A or D or stop [float * 3](one-hot encoding)
    Space: [bool]
    Ctrl: [bool]
    Shift: [bool]
    DelYaw: [float]
    DelPitch: [float]
    Attack: [float](only in discriminator)
    
    image (array) by pyautogui
    
    output
    WSmove: W or S or stop [float * 3](one-hot encoding)
    ADmove: A or D or stop [float * 3](one-hot encoding)
    Space: [bool]
    Ctrl: [bool]
    Shift: [bool]
    DelYaw: [float]
    DelPitch: [float]
    Attack: [float]
'''