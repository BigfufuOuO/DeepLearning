import torch
from Convolutional import ImagesClassifierModel
from DataLoad import load_data
from Model import train_model
from Plots import PlotsAndLogs
import time

num_epochs = 30
learning_rate = 0.001
weight_decay = 0.001

model = ImagesClassifierModel(kernel_size=3, padding=1, dropout=0.25)
train_loader, val_loader, test_loader = load_data(batch_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
milestone = [5]
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, 
                                                       threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
ploter = PlotsAndLogs()

start_time = time.perf_counter()
train_losses, val_losses, train_accuracies, val_accuracies, true_label, predicted_label, learning_rate_array = train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs)
end_time = time.perf_counter()
ploter.plot_loss_accuracy_lr(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies, learning_rate_array)
ploter.plot_acc_matrix(true_label, predicted_label)
ploter.caculate_class_accuracy(true_label, predicted_label)
ploter.output_logs(num_epochs, learning_rate, model, 
                   train_losses, val_losses, train_accuracies, val_accuracies, 
                   learning_rate_array, scheduler,
                   end_time - start_time)
if val_accuracies[-1] > 0.8:
    accurate_name = int(val_accuracies[-1] * 10000)
    torch.save(model.state_dict(), f'{ploter.Model_fileplace}/model_{accurate_name}.pth')
    print("Model has been saved.")
print("Completely Done!")

