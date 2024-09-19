import matplotlib.pyplot as plt
import pandas as pd
loss_path = './model/loss'
data = pd.read_csv(loss_path + '/gloss.csv')
data2 = pd.read_csv(loss_path + '/dloss.csv')
whole_g_loss = list(data['Generator Loss'])
whole_d_loss = list(data2['Discriminator Loss'])
index = [i + 1 for i in range(len(whole_g_loss))]
index2 = [(i + 1) * len(whole_g_loss) // len(whole_d_loss) for i in range(len(whole_d_loss))]
plt.plot(index[:], whole_g_loss[:], label='Generator Loss')
plt.plot(index2[:], whole_d_loss[:], label='Discriminator Loss')
plt.legend()
plt.show()