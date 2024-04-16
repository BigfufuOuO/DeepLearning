import sys
import time
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']

class PlotsAndLogs:
    def __init__(self):
        if sys.platform.startswith('win'):
            self.Model_fileplace = f'ConvolutionalNN/Model'
            self.fig_fileplace = f'ConvolutionalNN/figs'
            self.log_fileplace = f'ConvolutionalNN/logs'
        else:
            self.Model_fileplace = f'./ConvolutionalNN/Model'
            self.fig_fileplace = f'./ConvolutionalNN/figs'
            self.log_fileplace = f'./ConvolutionalNN/logs'
        self.daytime = time.strftime('%m%d-%H%M')
        self.acc_table = [[0] * 10 for _ in range(10)]
        self.class_num = [0] * 10
    
    def plot_loss_accuracy_lr(self, num_epoch, train_losses, val_losses, train_acc, val_acc, lr):
        plt.figure(figsize=(12, 12))
        epoch = range(1, num_epoch + 1) #array range from 1 to num_epoch
        plt.subplot(2, 2, 1)
        plt.plot(epoch, train_losses, label='Training Loss', color='blue', alpha=0.8)
        plt.plot(epoch, val_losses, label='Validation Loss', color='orange', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(epoch, train_acc, label='Training Accuracy', color='blue', alpha=0.8)
        plt.plot(epoch, val_acc, label='Validation Accuracy', color='orange', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(epoch, lr, label='Learning Rate', color='blue', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Decay')
        
        
        plt.savefig(f'{self.fig_fileplace}/loss_accuracy_{self.daytime}.png')
        
    def plot_acc_matrix(self, true_label, predicted_label, mode='validation'):
        cm = confusion_matrix(true_label, predicted_label)
        display = ConfusionMatrixDisplay(cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        display.plot(xticks_rotation='vertical', ax=ax)
        plt.savefig(f'{self.fig_fileplace}/confusion_matrix_{mode}_{self.daytime}.png')
        plt.show()
        
    def caculate_class_accuracy(self, true_label, predicted_label):
        for i in range(len(true_label)):
            self.acc_table[true_label[i]][predicted_label[i]] += 1
            self.class_num[true_label[i]] += 1
        
        
    def output_logs(self, num_epochs, learning_rate, model,
                    train_losses, val_losses, train_accuracies, val_accuracies, lr, scheduler, spend_time):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(f'{self.log_fileplace}/logs.txt', 'a') as f:
            f.write(f'Time: {self.daytime}, device:{device}\n')
            f.write(f'Time spent: {spend_time:.2f} seconds, ')
            f.write(f'Number of epochs: {num_epochs}, ')
            f.write(f'Learning rate: {learning_rate}, \n')
            f.write(f'Model kernel size: {model.kenel_size}, ')
            f.write(f'Model padding: {model.padding}, ')
            f.write(f'Model dropout: {model.dropout}\n')
            f.write(f'train_losses: {[f"{loss:.4f}" for loss in train_losses]}\n')
            f.write(f'val_losses: {[f"{loss:.4f}" for loss in val_losses]}\n')
            f.write(f'train_accuracies: {[f"{acc:.4f}" for acc in train_accuracies]}\n')
            f.write(f'val_accuracies: {[f"{acc:.4f}" for acc in val_accuracies]}\n')
            formated_lr = ["{}".format(rate) for rate in lr]
            f.write(f'learning_rate: {formated_lr}\n')
            f.write(f'scheduler: {type(scheduler).__name__}\n')
            total_acc = 0
            total_val = 0
            for label in labels:
                i = labels.index(label)
                accuracy = self.acc_table[i][i] / self.class_num[i]
                f.write(f'{label}({self.class_num[i]}): {self.acc_table[i]}, accuracy: {accuracy:.4f}\n')
                total_val += self.class_num[i]
                total_acc += self.acc_table[i][i]
            f.write('---------------VALIDATION----------------\n')
            f.write(f'Total acc: {total_acc / total_val}, Num of data: {total_val}\n')
            f.write(f'Model network: {model.network}\n')
            f.write('------------------------------------\n')
        
        
    
        