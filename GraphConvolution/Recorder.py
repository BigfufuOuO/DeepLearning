import os
import time
import matplotlib.pyplot as plt

class Recorder:
    def __init__(self):
        # obtain the current path
        current_path = os.path.abspath(os.path.dirname(__file__))
        self.fig_path = os.path.join(current_path, 'figs')
        self.log_path = os.path.join(current_path, 'logs')
        self.daytime = time.strftime('%m%d-%H%M')
    
    def Plot_loss(self, epochs, array_train_loss, array_val_loss, array_train_acc, array_val_acc, origial_dataset_name):
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
        
        plt.savefig(f'{self.fig_path}/{origial_dataset_name}_loss_acc_{self.daytime}.png')
        
    def Output_logs(self, array_train_loss, array_val_loss, array_train_acc, array_val_acc, orginal_dataset_name):
        '''
        Input:
            array_train_loss: list, the list of training loss
            array_val_loss: list, the list of validation loss
            array_train_acc: list, the list of training accuracy
            array_val_acc: list, the list of validation accuracy
            orginal_dataset_name: str, the name of dataset
        '''
        with open(f'{self.log_path}/{orginal_dataset_name}_logs.txt', 'a') as f:
            f.write('----------------------------------------------------------\n')
            f.write(f'Time: {self.daytime}\n')
            # each 10 record turn a new line
            f.write('Training Loss: \n')
            for i in range(len(array_train_loss)):
                f.write(f'{array_train_loss[i]:.4f}, ')
                if i % 10 == 9:
                    f.write(f'**Epoch: {i+1}\n')
            
            f.write('Validation Loss: \n')
            for i in range(len(array_val_loss)):
                f.write(f'{array_val_loss[i]:.4f}, ')
                if i % 10 == 9:
                    f.write(f'**Epoch: {i+1}\n')
                    
            f.write('Training Accuracy: \n')
            for i in range(len(array_train_acc)):
                f.write(f'{array_train_acc[i]:.4f}, ')
                if i % 10 == 9:
                    f.write(f'**Epoch: {i+1}\n')
                    
            f.write('Validation Accuracy: \n')
            for i in range(len(array_val_acc)):
                f.write(f'{array_val_acc[i]:.4f}, ')
                if i % 10 == 9:
                    f.write(f'**Epoch: {i+1}\n')
                    
    def Output_parameters(self, learning_rate, weight_decay, drop_rate, epochs, dataset_name):
        '''
        Input:
            learning_rate: float, the learning rate
            weight_decay: float, the weight decay
            epochs: int, the number of epochs
            dataset_name: str, the name of dataset
        '''
        with open(f'{self.log_path}/{dataset_name}_logs.txt', 'a') as f:
            f.write(f'Learning rate: {learning_rate}\n')
            f.write(f'Weight decay: {weight_decay}\n')
            f.write(f'Drop rate: {drop_rate}\n')
            f.write(f'Epochs: {epochs}\n')
            f.write('--------------------------END------------------------------\n')