from GraphConvolution.DataLoad import load_data
import matplotlib.pyplot as plt
import numpy as np

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']

# Define the dimensions of the plot grid 
W_grid = 5
L_grid = 5

train_loader, val_loader, test_loader = load_data()
def plot_images(set=train_loader):
    fig, axes = plt.subplots(L_grid, W_grid, figsize = (10,10))
    axes = axes.ravel() # flatten
    images, label = next(iter(set)) # get some samples from the train data

    for i in np.arange(0, W_grid * L_grid): 
        image = images[i]
        image = image / 2 + 0.5 # unnormalize
        npimg = image.numpy() # convert to numpy array
        axes[i].imshow(np.transpose(npimg, (1, 2, 0)))
        axes[i].set_title(labels[label[i].item()])
        axes[i].axis('off')
        
    plt.subplots_adjust(hspace=0.4)
    plt.show()

plot_images(train_loader)