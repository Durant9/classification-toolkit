import torch
import numpy as np

def train_model(classifier, optimizer, loader, criterion, device):
    '''
    Train the model for an epoch.
    Inputs: 
    - classifier: the instantiated model
    - optimizer: the optimizer linked to the classifier weights
    - loader: train dataloader
    - criterion: loss used
    - device: working device

    Outputs: 
    - classifier: trained model
    - optimizer: updated optimizer
    - train losses mean during the epoch
    '''
    train_losses = []
    for images, labels in loader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return classifier, optimizer, np.mean(train_losses)


def validate_model(classifier, loader, criterion, device):
    '''
    Validate the model through all a loader.
    Inputs:
    - classifier: the trained model
    - loader: the validation data loader
    - criterion: used loss
    - device: working device

    outputs:
    - mean of the obtained losses
    '''
    val_losses = []
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        val_losses.append(loss.item())

        _, preds = torch.max(torch.softmax(outputs, dim=1), dim=1)
        all_preds.append(preds)
        all_labels.append(labels)
    return np.mean(val_losses), all_preds, all_labels


def test_model(classifier, loader, device):
    '''
    Test the model.
    Inputs:
    - classifier: trained model
    - loader: test dataloader
    - device: working device

    Outputs:
    - all_images: list with all the PIL.Image.Image extracted from the loader
    - preds: list with all the model class prediction on the images
    - labels: list with all the actual labels of each image
    - confs: list with the confidence level of the model for each prediction
    '''
    all_images = []
    all_preds = []
    all_labels = []
    all_confs = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = classifier(images)
        confs, preds = torch.max(torch.softmax(outputs, dim=1), dim=1)
        all_images.extend(images.cpu())
        all_preds.append(preds)
        all_labels.append(labels)
        all_confs.append(confs)
    preds = torch.cat(all_preds).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    confs = torch.cat(all_confs).cpu().numpy()
    return all_images, preds, labels, confs
