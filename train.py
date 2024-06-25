import torch
import torch.nn as nn
import model
import os
import pyautogui
import BodyDetect
import socket
import json
import server
import numpy
#from threading import Thread

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

stat = torch.zeros((9, 4))
vel = torch.zeros((9, 4))
oppopos = torch.zeros((9, 6, 2))
oppogrid = torch.zeros((9, 6))
def generate(msg):
    global stat
    global vel
    global oppopos
    global oppogrid

    stat = torch.cat((stat, torch.tensor([[msg['ai']['isOnDamage'], msg['ai']['isOnGround'], msg['ai']['isSneaking'], msg['ai']['isSprinting']]])))
    vel = torch.cat((vel, torch.tensor([[msg['ai']['pitch'], *msg['ai']['velocity']]])))

    img = pyautogui.screenshot()
    pos, grid = BodyDetect.detection(numpy.array(img))
    oppopos = torch.cat((oppopos, torch.tensor([pos])))
    oppogrid = torch.cat((oppogrid, torch.tensor([grid])))

    gen = generator(stat.unsqueeze(dim=0), vel.unsqueeze(dim=0), oppogrid.unsqueeze(dim=0), oppopos.unsqueeze(dim=0))
    stat = stat[1:-1]
    vel = vel[1:-1]
    oppopos = oppopos[1:-1]
    oppogrid = oppopos[1:-1]

    return {"rotation" : gen[0], "velocity" : gen[1], "isSneaking" : gen[2], "isSprinting" : gen[3], "attackIndex" : (gen[4] >= torch.FloatTensor([0.5]).to(device)) - 1}
def save():
    torch.save(discriminator.state_dict(), disc_path + '/discriminator.pt')
    torch.save(generator.state_dict(), gen_path + '/generator.pt')

'''def Total(msg):
    th1 = Thread(target = generate, args=(msg))'''

server.loop(generate, save)