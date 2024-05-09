from Plots import PlotsAndLogs
from Convolutional import ImagesClassifierModel
import torch
import numpy as np
from GraphConvolution.DataLoad import load_data

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']

model = ImagesClassifierModel(kernel_size=3, padding=1, dropout=0.25)
ploter = PlotsAndLogs()
#ploter.output_model_architecture(model)
criterion = torch.nn.CrossEntropyLoss()

train_loader, val_loader, test_loader = load_data(batch_size=64)

model.load_state_dict(torch.load(ploter.Model_fileplace + '/model_8515.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(ploter.Model_fileplace + '/model_8578.pth', map_location=device))
model = model.to(device)
model.eval()

with torch.no_grad():
    num_inaccurate_test = 0
    true_label = []
    predicted_label = []
    for inputs, labels in test_loader:
        test_loss = []
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        output = model(inputs)
        _, predicted = torch.max(output, 1)
        test_inaccurate = model.inaccuracy(labels, predicted)
        num_inaccurate_test += test_inaccurate
        
        loss = criterion(output, labels)
        
        test_loss.append(loss.item())
        for i in range(len(labels)):
            true_label.append(labels[i].item())
            predicted_label.append(predicted[i].item())
    
    test_losses_mean = np.mean(test_loss)
    test_accuracy = 1 - num_inaccurate_test / len(test_loader.dataset)
    
ploter.plot_acc_matrix(true_label, predicted_label, 'test')
ploter.plot_test_images(test_loader, true_label, predicted_label)
ploter.caculate_class_accuracy(true_label, predicted_label)
total_acc = 0
total_val = 0
for label in classes:
    i = classes.index(label)
    accuracy = ploter.acc_table[i][i] / ploter.class_num[i]
    print(f'{label}({ploter.class_num[i]}): {ploter.acc_table[i]}, accuracy: {accuracy:.4f}')
    total_val += ploter.class_num[i]
    total_acc += ploter.acc_table[i][i]
print(f'Total acc: {total_acc / total_val}, Num of data: {total_val}')
print('Test complete.')