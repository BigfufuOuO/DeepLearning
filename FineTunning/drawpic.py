import matplotlib.pyplot as plt
import time

# 画loss曲线以及accuracy曲线
def draw_loss_curve(loss_list, acc_list, num_epochs):
    plt.figure()
    plt.plot(range(num_epochs), loss_list, label="loss")
    plt.plot(range(num_epochs), acc_list, label="accuracy")
    plt.legend()
    plt.savefig(f'./figs/loss_curve_{time.strftime("%Y%m%d-%H%M", time.localtime())}.png')
    
def draw_lr(lr_list, num_epochs):
    plt.figure()
    plt.plot(range(num_epochs), lr_list, label="lr")
    plt.legend()
    plt.savefig(f'./figs/lr_curve_{time.strftime("%Y%m%d-%H%M", time.localtime())}.png')