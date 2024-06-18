import torch
import torch.nn as nn
import model
import os
import pyautogui
import BodyDetect
import socket
import json
import server

dropout = 0
inputsize = 32
lstmin = 128
hidden = 1024
outhidden = 512
detecthid = 512
DiscStep = 0
disc_path = './model/disc'
gen_path = './model/gen'

learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.BCELoss().to(device)
EncGen = model.EncGen(dropout, device, inputsize, lstmin, hidden)
generator = model.ManyToOne(EncGen, hidden, outhidden).to(device)

if 'generator.pt' in os.listdir(gen_path):
    generator = torch.load(gen_path + '/generator.pt')

discriminator = model.discriminator(dropout, device, inputsize, lstmin, hidden, detecthid)
if 'discriminator.pt' in os.listdir(disc_path):
    discriminator = torch.load(disc_path + '/discriminator.pt')

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

def train(msg):
    global DiscStep
    DiscStep += 1

    stat = torch.tensor([msg['isOnDamage'], msg['isOnGround'], msg['isSneaking'], msg['isSprinting']])
    vel = torch.tensor([msg['pitch'], *msg['velocity']])

    img = pyautogui.screenshot()
    oppopos, oppogrid = BodyDetect.detection(img)

    gen = generator(stat, vel, oppogrid, oppopos)

    return {"rotation" : gen[0], "velocity" : gen[1], "isSneaking" : gen[2], "isSprinting" : gen[3], "attackIndex" : (gen[4] >= torch.FloatTensor([0.5]).to(device)) - 1}
def save():
    torch.save(discriminator.state_dict(), disc_path + '/discriminator.pt')
    torch.save(generator.state_dict(), gen_path + '/generator.pt')

server.loop(train, save)