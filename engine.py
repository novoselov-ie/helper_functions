import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
# Function to create the directory for saving weights
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')


# Loss calculation per batch
def loss_batch(loss_func, model, data, target, device, optimizer=None):
    inputs, targets = data.to(device), target.to(device)

    if optimizer is not None:
        optimizer.zero_grad()

    outputs = model(inputs)
    loss = loss_func(outputs, targets)

    if optimizer is not None:
        loss.backward()
        optimizer.step()

    return loss.item(), len(data)

# Loss calculation per epoch
def loss_epoch(loss_func, model, dataloader, device, optimizer=None):
    model.eval() if optimizer is None else model.train()

    running_loss = 0.0
    num_samples = 0

    for data, target in tqdm(dataloader, desc='Batches', leave=False):
        loss, batch_size = loss_batch(loss_func, model, data, target, device, optimizer)
        running_loss += loss
        num_samples += batch_size

    return running_loss / num_samples

# Training function
def train_regression_model(model, params):
    writer = SummaryWriter()
    num_epochs = params['num_epochs']
    optimizer = params['optimizer']
    lr_scheduler = params['lr_scheduler']
    path2weights = params['path2weights']
    device = params['device']
    loss_func = params['loss_func']
    train_loader = params['train_dl']
    valid_loader = params['val_dl']
    loss_history = {'train': [], 'val': []}

    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{num_epochs}, current lr={current_lr}')

        model.train()
        train_loss = loss_epoch(loss_func, model, train_loader, device, optimizer)
        loss_history['train'].append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(loss_func, model, valid_loader, device)
        loss_history['val'].append(val_loss)

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            print('Get best val_loss')
            torch.save(model.state_dict(), path2weights)

        lr_scheduler.step(val_loss)

        print(f'train loss: {train_loss*100:.4f}, val loss: {val_loss*100:.4f}, time: {(time.time() - start_time) / 60:.4f} min')
        print('-' * 100)

    # Close the TensorBoard writer
    writer.close()

    return model, loss_history
