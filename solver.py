import numpy as np 
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

def train(
        model,
        criterion,
        optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
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
            - `max_epoch`: Maximum Epoch for training
            - `max_non_increasing_epoch_count`: For early termination of model training
    """
    avg_train_loss, avg_train_acc = [], []
    avg_val_loss, avg_val_acc = [], []
    
    non_increasing_epoch_count = 0

    for epoch in range(max_epoch):
        epoch_train_loss, epoch_train_acc = __train_one_epoch(model, train_dataloader, optimizer, criterion, epoch)
        epoch_val_loss, epoch_val_acc = __validate(model, criterion, val_dataloader, epoch)
        
        avg_train_loss.append(epoch_train_loss)
        avg_train_acc.append(epoch_train_acc)
        avg_val_loss.append(epoch_val_loss)
        avg_val_acc.append(epoch_val_acc)

        # Check if epoch validation accuracy score is increasing.
        # Stop model training when validation accuracy does not increase for a few epochs
        if len(avg_val_acc) > 1 and avg_val_acc[-1] <= avg_val_acc[-2]:
            non_increasing_epoch_count += 1
            if non_increasing_epoch_count >= max_non_increasing_epoch_count:
                break
    
    return model, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc

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
