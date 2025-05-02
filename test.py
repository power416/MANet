import os
import matplotlib.pyplot as plt
import numpy as np
from models import *
from pylab import *
from obspy import read
from torch.backends import cudnn
from function import *
import warnings
warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if torch.cuda.is_available():
    device = 'cuda'
    cudnn.benchmark = True
else:
    device = 'cpu'

model_dir = './snapshots/best_model.pt'
# year = '2019'
path = './data/'

data_dirs1 = os.listdir(path)
data_dirs2 = [os.path.join(path, dirs) for dirs in data_dirs1]
g = 0
no_quake = []
threshold_p = 0.3
threshold_s = 0.3
# threshold_d=0.5

samplerate = 10
k = 600 * samplerate
slide_step = k//8
num1 = 100 * samplerate
num2 = 300 * samplerate

color1 = plt.cm.viridis(0.25)
color2 = plt.cm.plasma(0.5)

for data_dir in data_dirs2:
    if 'none.npy' not in data_dir:
        year = data_dir.split('/')[-1].split('.')[-3]
        name = data_dir.split('/')[-1].split('.')[-2].split('_')[0]

        path2 = './results/%s/' % year
        if not os.path.exists(path2):
            os.makedirs(path2)
        output_dir1 = path2 + '%s/global/' % name
        output_dir2 = path2 + '%s/local/' % name
        if not os.path.exists(output_dir1):
            os.makedirs(output_dir1)
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)

        data_name = os.path.join(path, data_dir.split('/')[-1])

        st = read(data_name)
        print(st)
        data, z_start_num, n_start_num, e_start_num = preprocessing(st)
  
        start_date = str(st[0].stats.starttime)[:10]
        if samplerate == 10:
            data = data[0::2, :]
        m, _ = data.shape
        num = (m - k) // slide_step + 1
        # print(num)
        
        ckp = torch.load(model_dir, map_location=device, weights_only=True)
        net = MAN()
        net = nn.DataParallel(net)
        net.load_state_dict(ckp)
        net.eval()

        # print(data2.shape)
        s_pmax = []
        s_smax = []
        r_b = []
        print('Beginning Global!')
        for i in range(int(num)):
            # data2 = np.zeros([k, 3])
            data_copy = data.copy()
            a = i * slide_step
            b = i * slide_step + k

            data2 = data_copy[a:b, :]
            data3 = data2.copy()

            # plt.plot(data2)
            # plt.show()
            data2 = normalize(data2)
            data2 = torch.from_numpy(data2).float()
            data2 = torch.permute(data2, [1, 0])
            data2 = torch.unsqueeze(data2, dim=0)
            data2 = data2.to(device)
            
            # print(data3.shape)

            x_d, x_p, x_s = net(data2)

            x_p = torch.squeeze(x_p, dim=0)
            x_p = torch.permute(x_p, [1, 0])
            x_p = x_p.cpu().detach().numpy()
            p_max = np.max(x_p)
            s_pmax.append(p_max)
            
            x_s = torch.squeeze(x_s, dim=0)
            x_s = torch.permute(x_s, [1, 0])
            x_s = x_s.cpu().detach().numpy()
            s_max = np.max(x_s)
            s_smax.append(s_max)

            if p_max >= threshold_p and s_max >= threshold_s and i>=10:
                r_p, _ = np.where(x_p == p_max)
                if r_p[0] - 20 < 0:
                    x_p[int(r_p[0] + 20):, :] = 0
                elif r_p[0] + 20 > k:
                    x_p[:int(r_p[0] - 20), :] = 0
                else:
                    x_p[:int(r_p[0] - 20), :] = 0
                    x_p[int(r_p[0] + 20):, :] = 0

                r_s, _ = np.where(x_s == s_max)
                if r_s[0] - 20 < 0:
                    x_s[int(r_s[0] + 20):, :] = 0
                elif r_s[0] + 20 > k:
                    x_s[:int(r_s[0] - 20), :] = 0
                else:
                    x_s[:int(r_s[0] - 20), :] = 0
                    x_s[int(r_s[0] + 20):, :] = 0

                if r_s[0] - r_p[0] >= num1 and r_s[0] - r_p[0] <= num2:
                    r_b.append(i)
                    nframes, _ = data3.shape
                                
                    time2 = np.arange(0, nframes) * (1.0/samplerate)

                    r_p1 = r_p[0] / int(samplerate)
                    r_s1 = r_s[0] / int(samplerate)

                    fig_time = start_date + 'T' + timee(z_start_num + (a/samplerate) * 1000) + '000Z'
                    P_time = timee(z_start_num + (a + int(r_p[0]))/samplerate * 1000)
                    S_time = timee(z_start_num + (a + int(r_s[0]))/samplerate * 1000)

                    fig = plt.figure(figsize=(8, 5))
                    ax = fig.add_subplot(411)
                    plt.plot(time2, data3[:, 0], 'k')
                    # legend_properties = {'weight': 'bold'}
                    plt.tight_layout()
                    ymin, ymax = ax.get_ylim()
                    pl = plt.vlines(r_p1, ymin, ymax, color=color1, linewidth=2, label='Predicted P-arrival')
                    sl = plt.vlines(r_s1, ymin, ymax, color=color2, linewidth=2, label='Predicted S-arrival')
                    # plt.legend(handles=[pl, sl], loc='upper right', borderaxespad=0., prop=legend_properties)
                    ax.set_xticklabels([])

                    ax = fig.add_subplot(412)
                    plt.plot(time2, data3[:, 1], 'k')
                    # legend_properties = {'weight': 'bold'}
                    plt.tight_layout()
                    ymin, ymax = ax.get_ylim()
                    pl = plt.vlines(r_p1, ymin, ymax, color=color1, linewidth=2, label='Predicted P-arrival')
                    sl = plt.vlines(r_s1, ymin, ymax, color=color2, linewidth=2, label='Predicted S-arrival')
                    # plt.legend(handles=[pl, sl], loc='upper right', borderaxespad=0., prop=legend_properties)
                    ax.set_xticklabels([])

                    ax = fig.add_subplot(413)
                    plt.plot(time2, data3[:, 2], 'k')
                    # legend_properties = {'weight': 'bold'}
                    plt.tight_layout()
                    ymin, ymax = ax.get_ylim()
                    pl = plt.vlines(r_p1, ymin, ymax, color=color1, linewidth=2, label='Predicted P-arrival')
                    sl = plt.vlines(r_s1, ymin, ymax, color=color2, linewidth=2, label='Predicted S-arrival')
                    # plt.legend(handles=[pl, sl], loc='upper right', borderaxespad=0., prop=legend_properties)
                    ax.set_xticklabels([])

                    ax = fig.add_subplot(414)
                    legend_properties = {'weight': 'bold'}
                    plt.plot(time2, x_p, color=color1)
                    plt.plot(time2, x_s, color=color2)
                    plt.legend(handles=[pl, sl], loc='upper right', borderaxespad=0., prop=legend_properties)
                    plt.tight_layout()
                    plt.xlabel('time (s) after %s,   P arrival time: %s,   S arrival time: %s' % (fig_time, P_time, S_time), fontsize=9)
                    plt.savefig(output_dir2 + 'p[%s]_s[%s].png' % (str(a + r_p[0]), str(a + r_s[0])), dpi=300)
                    plt.close()

            print('\rFile: [%d / %d]   |Windows: [%d / %d]   |year: %s   | name: %s' % (g+1, len(data_dirs2), i, int(num), year, name),
                  end='', flush=True)

        print('\nGlobal Complete!')
        np.save(path2 + name + '/pmaxs.npy', s_pmax)
        np.save(path2 + name + '/smax.npy', s_smax)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(s_pmax)
        plt.plot(s_smax)
        plt.axhline(threshold_p, color=color1, linestyle='--')
        plt.axhline(threshold_s, color=color2, linestyle='--')
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.xlabel('num', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.savefig(output_dir1 + 'global.png', dpi=300)
        plt.close()
        print('Save Complete!')

        if r_b == []:
            no_name = year + '_' + name
            no_quake.append(no_name)
            print('%s no quake event!' % no_name)
        g += 1
        print('\nLocal Complete!')
