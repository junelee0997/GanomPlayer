import torch
import torch.nn as nn
import model
import os
import pyautogui
import BodyDetect
import server
import numpy
import pygetwindow as gw
import matplotlib as plt

dropout = 0
player_count = 1
inputsize = 32
lstmin = 128
hidden = 1024
outhidden = 512
detecthid = 512
DiscStep = 0
time = 1
disc_path = './model/disc'
gen_path = './model/gen'

learning_rate = 0.01
device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.BCELoss().to(device)
EncGen = model.EncGen(dropout, device, inputsize, lstmin, hidden)
generator = model.ManyToOne(EncGen, hidden, outhidden).to(device)

if 'generator.pt' in os.listdir(gen_path):
    generator.load_state_dict(torch.load(gen_path + '/generator.pt'))

discriminator = model.discriminator(dropout, device, inputsize, lstmin, hidden, detecthid)
if 'discriminator.pt' in os.listdir(disc_path):
    discriminator.load_state_dict(torch.load(disc_path + '/discriminator.pt'))

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

stat = torch.zeros((9, 4), device=device)
vel = torch.zeros((9, 4), device=device)
oppopos = torch.zeros((9, 6, 2), device=device)
oppogrid = torch.zeros((9, 6), device=device)

plstat = [torch.zeros((9, 4), device=device) for i in range(player_count)]
plvel = [torch.zeros((9, 4), device=device) for i in range(player_count)]

resvel = torch.zeros((9, 4), device=device)

stat.requires_grad_(True)
vel.requires_grad_(True)
oppogrid.requires_grad_(True)
oppopos.requires_grad_(True)
def generate(msg, client_socket):
    global stat, vel, plstat, plvel, resvel, oppopos, oppogrid, DiscStep, g_optimizer, d_optimizer

    g_optimizer.zero_grad()
    stat = torch.cat((stat, torch.tensor([[msg['ai']['isOnDamage'], msg['ai']['isOnGround'], msg['ai']['isSneaking'], msg['ai']['isSprinting']]], device=device)))
    vel = torch.cat((vel, torch.tensor([[msg['ai']['pitch'], *msg['ai']['velocity']]], device=device)))
    for i in range(player_count):
        plstat[i] = torch.cat((plstat[i], torch.tensor(
            [[msg['players'][i]['isOnDamage'], msg['players'][i]['isOnGround'], msg['players'][i]['isSneaking'], msg['players'][i]['isSprinting']]], device=device)))
        plvel[i] = torch.cat((plvel[i], torch.tensor([[msg['players'][i]['pitch'], *msg['players'][i]['velocity']]], device=device)))

    DiscStep += 1
    if DiscStep == time:
        d_optimizer.zero_grad()

    window = gw.getWindowsWithTitle("Minecraft 1.8.9")[0]
    img = pyautogui.screenshot(r'C:\Users\Administrator\Desktop\GanomPlayer\model\test.png',
                               region=(window.left, window.top, window.width, window.height))
    pos, grid = BodyDetect.detection(numpy.array(img))
    oppopos = torch.cat((oppopos, torch.tensor([pos], device=device)))
    oppogrid = torch.cat((oppogrid, torch.tensor([grid], device=device)))


    gen = generator(stat, vel, oppogrid, oppopos)

    vel = vel[1:]

    oppopos = oppopos[1:]
    oppogrid = oppogrid[1:]
    resvel = torch.cat((resvel,torch.cat((torch.tensor([gen[0][1]], device=device), torch.tensor([gen[1][0].item(), float(gen[5] >= 0.5), gen[1][1].item()], device=device))).unsqueeze(dim = 0))) # y속도 0
    server.send({"rotation" : gen[0].tolist(), "velocity" : [gen[1][0].item(), float(gen[5] >= 0.5), gen[1][1].item()], "isSneaking" : gen[2].item() >= 0.5, "isSprinting" : gen[3].item() >= 0.5, "attackIndex" : (int(gen[4] >= 0.5) - 1)}, client_socket)

    g_loss = criterion(discriminator(stat.detach(), resvel.detach()), torch.tensor([0], device=device).to(torch.float32).requires_grad_(True))

    stat = stat[1:]
    resvel = resvel[1:]

    if DiscStep == time:
        r_loss = 0
        for i in range(player_count):
            r_loss += criterion(discriminator(plstat[i].detach(), plvel[i].detach()), torch.tensor([1], device=device).to(torch.float32).requires_grad_(True))
        d_loss = (g_loss + r_loss) / (1 + player_count)
        d_loss.backward(retain_graph=True)
        d_optimizer.step()
        DiscStep = 0
        print(g_loss, d_loss)

    plstat = [plstat[i][1:] for i in range(player_count)]
    plvel = [plvel[i][1:] for i in range(player_count)]

    g_loss.backward(retain_graph=True)
    g_optimizer.step()

def save():
    torch.save(discriminator.state_dict(), disc_path + '/discriminator.pt')
    torch.save(generator.state_dict(), gen_path + '/generator.pt')


server.loop(generate, save)