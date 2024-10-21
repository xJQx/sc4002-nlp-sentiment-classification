import os
import numpy as np 
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.file import _makeDir_if_does_not_exist

def train(
        model,
        criterion,
        optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        min_epoch: int = 20,
        max_epoch: int = 10_000,
        max_non_increasing_epoch_count: int = 10
    ):
    """
        Model training step.
        Training will stop when validation accuracy score is not increasing for `max_non_increasing_epoch_count` epochs,
        or when `max_epoch` is reached

        ### Parameters
            - `model`: Model to be trained
            - `criterion`: Loss function
            - `optimizer`: Optimizer function
            - `train_dataloader`: Dataset for training
            - `val_dataloader`: Dataset for validation
            - `min_epoch`: Minimum Epoch for training
            - `max_non_increasing_epoch_count`: For termination of model training
    """
    avg_train_loss, avg_train_acc = [], []
    avg_val_loss, avg_val_acc = [], []
    
    non_increasing_epoch_count = 0
    num_of_epochs = 0

    for epoch in range(1, max_epoch+1):
        num_of_epochs += 1

        epoch_train_loss, epoch_train_acc = __train_one_epoch(model, train_dataloader, optimizer, criterion, epoch)
        epoch_val_loss, epoch_val_acc = __validate(model, criterion, val_dataloader, epoch)
        
        avg_train_loss.append(epoch_train_loss)
        avg_train_acc.append(epoch_train_acc)
        avg_val_loss.append(epoch_val_loss)
        avg_val_acc.append(epoch_val_acc)

        # Check if epoch validation accuracy score is increasing.
        # Stop model training when validation accuracy does not increase for a few epochs (starting from 20 epochs onwards)
        window_size = max(non_increasing_epoch_count, max_non_increasing_epoch_count)
        recent_highest_seen = max(avg_val_acc[-2: -2-window_size: -1]) if num_of_epochs >= window_size else 0 # Maximum validation accuracy seen in the recent window
        if num_of_epochs >= min_epoch and avg_val_acc[-1] <= recent_highest_seen:
            non_increasing_epoch_count += 1
            if non_increasing_epoch_count >= max_non_increasing_epoch_count:
                # Ensure best weights are returned
                if avg_val_acc[-1] == recent_highest_seen:
                    break
        else:
            non_increasing_epoch_count = 0
    
    return model, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, num_of_epochs

def __train_one_epoch(
        model,
        train_dataloader: DataLoader,
        optimizer,
        criterion,
        epoch=0
    ):
    model.train()
    
    total = len(train_dataloader)
    train_loss, train_acc = [], []
    total_step = 0
    
    with tqdm(total=total) as t:
        t.set_description('Epoch %i (Train)' % epoch)

        # Mini-Batch
        for i, (sentences, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predictions = model(sentences)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            # loss
            train_loss.append(loss.detach().numpy())

            # compute accuracy
            acc = __binary_accuracy(predictions, labels)
            train_acc.append(acc)
            
            total_step +=1
            t.set_postfix(loss=np.mean(train_loss), acc=np.mean(train_acc))
            t.update(1)
            
    return np.mean(train_loss), np.mean(train_acc)

def __binary_accuracy(predictions, labels):
    # Convert logits to predicted class labels (0 or 1)
    rounded_preds = torch.argmax(predictions, dim=1)
    
    # Compare predicted classes with actual labels
    correct = (rounded_preds == labels).float() # This will return 1 for correct predictions, 0 for incorrect
    
    # Calculate accuracy
    accuracy = correct.sum() / len(correct)
    
    return accuracy

def __validate(
        model, 
        criterion, 
        val_dataloader: DataLoader,
        epoch=0
    ):
    model.eval()
    
    total = len(val_dataloader)
    val_loss, val_acc = [], []
    total_step = 0
    
    with torch.no_grad():
        with tqdm(total=total) as t:
            t.set_description('Epoch %i (Val)' % epoch)
            for i, (sentences, labels) in enumerate(val_dataloader):
                predictions = model(sentences)
                
                # compute loss
                loss = criterion(predictions, labels)
                val_loss.append(loss.detach().numpy()) 
                
                # compute accuracy
                acc = __binary_accuracy(predictions, labels)
                val_acc.append(acc)
                
                total_step +=1
                t.set_postfix(loss=np.mean(val_loss), acc=np.mean(val_acc))
                t.update(1)
            
    return np.mean(val_loss), np.mean(val_acc)

def test(
        model,
        criterion, 
        test_dataloader: DataLoader
    ):
    model.eval()
    
    test_loss, test_acc = [], []
    
    with torch.no_grad():
        for i, (sentences, labels) in enumerate(test_dataloader):
            predictions = model(sentences)
            
            # compute loss
            loss = criterion(predictions, labels)
            test_loss.append(loss.detach().numpy()) 
            
            # compute accuracy
            acc = __binary_accuracy(predictions, labels)
            test_acc.append(acc)

    return np.mean(test_loss), np.mean(test_acc)


def plot_loss_acc_graph(
        loss_list: list[float], 
        acc_list: list[float],
        dataset_type: str,
        subtitle: str,
        save_filename_prefix: str = None,
        display: bool =True
    ):
    if save_filename_prefix.startswith("/"): save_filename_prefix = save_filename_prefix[1:]
    # Create save directory if does not exist
    if save_filename_prefix:
        dir_name = os.path.dirname(save_filename_prefix)
        if dir_name != "" and dir_name != "/":
            _makeDir_if_does_not_exist(f"plots/{dir_name}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    epoch_counts = [c for c in range(1, len(loss_list)+1)]

    # Overall Title for the figure
    fig.suptitle(f"({dataset_type}) {subtitle}", fontsize=16)

    # Loss
    ax1.plot(epoch_counts, loss_list)
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs Epoch Number")
    ax1.grid(True)

    # Accuracy
    ax2.plot(epoch_counts, acc_list, color="red")
    ax2.set_xlabel("Epoch Number")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Epoch Number")
    ax2.grid(True)
    
    if save_filename_prefix:
        plt.savefig(f"plots/{save_filename_prefix}_{dataset_type}_loss_accuracy.png")
    if display:
        plt.show()

    # Reset plot
    plt.close()
