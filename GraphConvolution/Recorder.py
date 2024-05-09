import matplotlib.pyplot as plt

class Recorder:
    def __init__(self) -> None:
        pass
    
    def Plot_loss(self, loss_list):
        '''
        Input:
            loss_list: list, the list of loss
        '''
        plt.plot(loss_list)
        plt.title('Loss')
        plt.show()