# Common Function Definitions

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm.notebook import tqdm

# Loss Function Value Calculation
def eval_loss(loader, device, net, criterion):

    # Retrieve the first set from the data loader
    for images, labels in loader:
        break

    # Send to GPU
    inputs = images.to(device)
    labels = labels.to(device)

    # Prediction calculation
    outputs = net(inputs)

    # Loss calculation
    loss = criterion(outputs, labels)

    return loss

# Learning Functions
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):

    base_epochs = len(history)

    for epoch in range(base_epochs, num_epochs+base_epochs):
        # Number of correct answers per epoch (for accuracy calculation)
        n_train_acc, n_val_acc = 0, 0

        # Cumulative loss per epoch (before averaging)
        train_loss, val_loss = 0, 0

        # Number of cumulative data records per epoch
        n_train, n_test = 0, 0

        # Training phase
        net.train()

        for inputs, labels in tqdm(train_loader):
            # Number of data records per batch
            train_batch_size = len(labels)

            # Number of data records accumulated per epoch
            n_train += train_batch_size

            # Transfer to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Initialize gradient
            optimizer.zero_grad()

            # 1. Calculate prediction
            outputs = net(inputs)

            # 2. Calculate loss
            loss = criterion(outputs, labels)

            # 3. Calculate gradient
            loss.backward()

            # 4. Adjust parameters
            optimizer.step()

            # Calculate the prediction label
            predicted = torch.max(outputs, 1)[1]

            # Calculate loss
            # Since the loss is calculated as an average, revert it to
            # the pre-averaging loss and add it back
            train_loss += loss.item() * train_batch_size

            # Calculate accuracy
            n_train_acc += (predicted == labels).sum().item()

        # Prediction phase
        net.eval()

        for inputs_test, labels_test in test_loader:
            # Number of data records per batch
            test_batch_size = len(labels_test)

            # Number of data records accumulated per epoch
            n_test += test_batch_size

            # Transfer to GPU
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # 1. Calculate prediction
            outputs_test = net(inputs_test)

            # 2. Calculate loss
            loss_test = criterion(outputs_test, labels_test)

            # Calculate the prediction label
            predicted_test = torch.max(outputs_test, 1)[1]

            # Calculate loss
            # Since the loss is calculated as an average, revert it to
            # the pre-averaging loss and add it back
            val_loss +=  loss_test.item() * test_batch_size

            # Calculate accuracy
            n_val_acc +=  (predicted_test == labels_test).sum().item()

        # Calculate the accuracy of the training and the validation
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test

        # Calculate the loss of the training and the validation
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test

        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])
        history = np.vstack((history, item))
    return history

# Learning Log Analysis
def evaluate_history(history):
    # Confirm the loss values and accuracies
    print(f'Initial: Loss: {history[0,3]:.5f} Accuracy: {history[0,4]:.5f}')
    print(f'Final: Loss: {history[-1,3]:.5f} Accuracy: {history[-1,4]:.5f}' )

    num_epochs = len(history)
    if num_epochs < 10:
        unit = 1
    else:
        unit = num_epochs / 10

    # Display the learning curve for the loss
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='Training')
    plt.plot(history[:,0], history[:,3], 'k', label='Validation')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('Repetition counts')
    plt.ylabel('Loss')
    plt.title('Learning curve of the Loss')
    plt.legend()
    plt.show()

    # Display learning curve for the accuracy
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='Training')
    plt.plot(history[:,0], history[:,4], 'k', label='Validation')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('Repetition counts')
    plt.ylabel('Accuracy')
    plt.title('Learning curve of the Accuracy')
    plt.legend()
    plt.show()

# PyTorch Random Number Fixing
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True