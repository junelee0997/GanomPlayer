import torch
import torch.nn as nn
import model
import os
import pyautogui
import server
import pygetwindow as gw
import pandas as pd
import cv2

whole_g_loss = []
whole_d_loss = []

dropout = 0
player_count = 1
inputsize = 32
lstmin = 128
hidden = 1024
outhidden = 512
detecthid = 512
DiscStep = 0
time = 10
disc_path = './model/disc'
gen_path = './model/gen'
loss_path = './model/loss'

learning_rate = 0.001
device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.BCELoss().to(device)
generator = model.Generator(device).to(device)

if 'generator.pt' in os.listdir(gen_path):
    generator.load_state_dict(torch.load(gen_path + '/generator.pt'))

discriminator = model.Discriminator(device).to(device)
if 'discriminator.pt' in os.listdir(disc_path):
    discriminator.load_state_dict(torch.load(disc_path + '/discriminator.pt'))

if 'gloss.csv' in os.listdir(loss_path):
    data = pd.read_csv(loss_path + '/gloss.csv')
    data2 = pd.read_csv(loss_path + '/dloss.csv')
    whole_g_loss = list(data['Generator Loss'])
    whole_d_loss = list(data2['Discriminator Loss'])

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

user_history = [torch.zeros((1, 9, 12)) for i in range(player_count)]
bot_history = torch.zeros((1, 9, 12))

user_img = [torch.zeros((1, 9, 140, 140, 3)) for i in range(player_count)]
bot_img = torch.zeros((1, 9, 140, 140, 3))
def generate(msg, client_socket):
    global DiscStep, g_optimizer, d_optimizer

    #g_optimizer.zero_grad()

    DiscStep += 1
    if DiscStep == time:
        d_optimizer.zero_grad()
    else:
        g_optimizer.zero_grad()

    window = gw.getWindowsWithTitle("Minecraft 1.8.9")[0] #bot
    img = pyautogui.screenshot(r'C:\Users\Administrator\Desktop\GanomPlayer\model\test.png',
                               region=(window.left, window.top, window.width, window.height))
    img = torch.tensor(img).unsqueeze(dim = 0).detach()
    img = cv2.resize(img, (140, 140))
    window = gw.getWindowsWithTitle("Minecraft 1.8.8")[0]  # bot
    img2 = pyautogui.screenshot(r'C:\Users\Administrator\Desktop\GanomPlayer\model\test.png',
                               region=(window.left, window.top, window.width, window.height))
    img2 = torch.tensor(img2).unsqueeze(dim=0).detach()
    img2 = cv2.resize(img2, (140, 140))

    #img, move1, move2, jmp, run, crh, del_yaw, del_pitch
    data = [torch.tensor(msg['ai']['WSmove']).unsqueeze(dim=0).detach(), torch.tensor(msg['ai']['ADmove']).unsqueeze(dim=0).detach(), torch.tensor(msg['ai']['Space']).unsqueeze(dim=0).detach(), torch.tensor(msg['ai']['Ctrl']).unsqueeze(dim=0).detach(), torch.tensor(msg['ai']['Shift']).unsqueeze(dim=0).detach(),torch.tensor(msg['ai']['DelYaw']).unsqueeze(dim=0).detach(), torch.tensor(msg['ai']['DelPitch']).unsqueeze(dim=0).detach(),torch.tensor(msg['ai']['Attack']).unsqueeze(dim=0).detach()]
    gen = generator(img, data[0], data[1], data[2], data[3], data[4], data[5], data[6])

    server.send({"WDmove" : gen[0].item(), "ADmove" : gen[1].item(), "Space" : gen[2].item() >= 0.5, "Ctrl" : gen[3].item() >= 0.5, "Shift" : gen[4].item() >= 0.5, "DelYaw" : gen[5].item(), "DelPitch" : gen[6].item(), "Attack" : gen[7].item() >= 0.5}, client_socket)
    #Gen = discriminator(stat.detach(), resvel.detach())

    '''if DiscStep == time:
        f_loss = criterion(Gen, torch.tensor([0], device=device).to(torch.float32).requires_grad_(True))
        r_loss = 0
        for i in range(player_count):
            r_loss += criterion(discriminator(plstat[i].detach(), plvel[i].detach()), torch.tensor([1], device=device).to(torch.float32).requires_grad_(True))
        d_loss = (f_loss + r_loss) / (1 + player_count)
        whole_d_loss.append(d_loss.item())
        d_loss.backward(retain_graph=True)
        d_optimizer.step()
        DiscStep = 0
        print("d_loss", d_loss)
    else:
        g_loss = criterion(Gen, torch.tensor([1], device=device).to(torch.float32).requires_grad_(True))
        whole_g_loss.append(g_loss.item())
        g_loss.backward(retain_graph=True)
        g_optimizer.step()
        print("g_loss", g_loss)'''

    #g_loss.backward(retain_graph=True)
    #g_optimizer.step()

def save():
    data = {'Generator Loss':whole_g_loss}
    data2 = {'Discriminator Loss': whole_d_loss}
    df = pd.DataFrame(data=data)
    df.to_csv(loss_path + '/gloss.csv', index=True)
    df2 = pd.DataFrame(data=data2)
    df2.to_csv(loss_path + '/dloss.csv', index=True)
    torch.save(discriminator.state_dict(), disc_path + '/discriminator.pt')
    torch.save(generator.state_dict(), gen_path + '/generator.pt')


server.loop(generate, save)