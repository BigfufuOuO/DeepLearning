import matplotlib.pyplot as plt

class Plots:
    def __init__(self):
        pass
    
    def plot_loss(self, num_epoch, train_losses, val_losses):
        epoch = range(1, num_epoch + 1) #array range from 1 to num_epoch
        plt.plot(epoch, train_losses, label='Training Loss', color='blue', alpha=0.5)
        plt.plot(epoch, val_losses, label='Validation Loss', color='orange', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend
        plt.show()
        
    def plot_accuracy(self, num_epoch, train_accuracies, val_accuracies):
        epoch = range(1, num_epoch + 1)
        plt.plot(epoch, train_accuracies, label='Training Accuracy', color='blue', alpha=0.5)
        plt.plot(epoch, val_accuracies, label='Validation Accuracy', color='orange', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()
        