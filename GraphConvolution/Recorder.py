import os
import time
import matplotlib.pyplot as plt

class Recorder:
    def __init__(self):
        # obtain the current path
        current_path = os.path.abspath(os.path.dirname(__file__))
        self.fig_path = os.path.join(current_path, 'figs')
        self.daytime = time.strftime('%m%d-%H%M')
    
    def Plot_loss(self, epochs, array_train_loss, array_val_loss, array_train_acc, array_val_acc):
        '''
        Input:
            loss_list: list, the list of loss
        '''
        plt.figure(figsize=(12, 12))
        epoch = range(1, epochs + 1)
        plt.subplot(2, 2, 1)
        plt.plot(epoch, array_train_loss, label='Training Loss', color='blue', alpha=0.8)
        plt.plot(epoch, array_val_loss, label='Validation Loss', color='orange', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(epoch, array_train_acc, label='Training Accuracy', color='blue', alpha=0.8)
        plt.plot(epoch, array_val_acc, label='Validation Accuracy', color='orange', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.savefig(f'{self.fig_path}/loss_accuracy_{self.daytime}.png')