import numpy as np 
import tqdm
import torch

def train(model, criterion, optimizer, train_set, val_set, max_epoch, batch_size):
    
    avg_train_loss, avg_train_acc = [], []
    avg_val_loss, avg_val_acc = [], []
    
    for epoch in range(max_epoch):
        epoch_train_loss = train_one_epoch(model, train_set, optimizer, criterion, epoch)
        epoch_val_loss, epoch_val_acc = validate(model, criterion, val_set, epoch)
        
        avg_train_loss.append(epoch_train_loss)
        avg_val_loss.append(epoch_val_loss)
        avg_val_acc.append(epoch_val_acc)
        
    
    return model, avg_train_loss, avg_val_loss, avg_val_acc

def train_one_epoch(model, train_set, optimizer, criterion, epoch=0):
    
    model.train()
    
    total = train_set.size()
    train_loss, train_acc = [], []
    total_step = 0
    
    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for inputs, labels in train_set:
            
            optimizer.zero_grad()
            predictions = model(inputs)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            # loss
            train_loss.append(loss.numpy())
            
            total_step +=1
            t.set_postfix(loss=np.mean(train_loss))
            t.update(1)
            
    return np.mean(train_loss)
            
            
            
            
    
    

def binary_accuracy(predictions, labels):
    rounded_preds = np.round(torch.sigmoid(predictions).numpy())
    correct = (rounded_preds == labels).astype(np.float32)
    accuracy = correct.sum() / len(correct)
    
    return accuracy

def validate(model, criterion, val_set, epoch=0):
    
    model.eval()
    
    total = val_set.size()
    val_loss, val_acc = [], []
    total_step = 0
    
    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for inputs, labels in val_set:
            predictions = model(inputs)
            
            # compute loss
            loss = criterion(predictions, labels)
            val_loss.append(loss.numpy()) 
            
            # compute accuracy
            acc = binary_accuracy(predictions, labels)
            val_acc.append(acc)
            
            total_step +=1
            t.set_postfix(loss=np.mean(val_loss), acc=np.mean(val_acc))
            t.update(1)
            
    return np.mean(val_loss), np.mean(val_acc)
        
    
        
    