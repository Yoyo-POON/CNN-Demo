import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np


    
class Reg_CNN(nn.Module):
    '''Build regularized model with batchnorm and dropout'''
    # Reuse regularized model build in Task1 
    # so that can have direct comparsion on the performance after using MixUp & Label Smoothing
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 2)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256, 32)
        self.dropout = nn.Dropout(0.15)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
def MixUp(img_batch, labels, alpha = 1.0):
    '''
    Function for Custom MixUp
    Parameters: 

    img_batch   sent the image data (x) to the function
    labels      sent the label data (y) to the function
    alpha       float, Default alpha = 1.0. set the alpha for labmda. labmda ~ Beta(alpha, alpha)

    Steps:
    1. get a ramdom labmda form Beta(alpha, alpha)
    2. randomly permutating the index to get the target pair image.  Make img pair up randomly.
    3. calculate the mixup img and label
    4. return img and labels
    '''
    Lambda = np.random.beta(alpha, alpha)

    shuffle_index = torch.randperm(img_batch.size(0))

    img = Lambda * img_batch + (1-Lambda) * img_batch[shuffle_index]
    labels = Lambda * labels + (1-Lambda) * labels[shuffle_index]

    return img, labels

def smoothing(labels, epsilon = 0.1):
    '''
    Function for Label Smoothing
    Parameters: 

    labels      sent the label data (y) to the function
    epsilon     float, Default epsilon = 0.1. between 0 to 1. set the epsilon value

    Steps:
    1. use setted epsilon to calculated smoothing labels
    4. return labels
    '''
    labels = ( 1 - epsilon) * labels + epsilon / 10 # 10 as number of classes
    return labels
    

def train(ModelClass, trainloader, valloader, weight_decay , fname):
    '''
    Function for Train the model with train set and validation set
    Then save the model

    Parameters: 

    ModelClass      model that want to train
    trainloader     sent the DataLoader from train set to the training loop
    valloader       sent the DataLoader from validation set to the training loop
    weight_decay    set weight decay for optimizer as a part of Regularisation
    fname           set the file name for save the model

    Steps:
    1. define traning setting
    2. Train the model with epochs loop
    3. Evaluate the model with validation set
    4. save the training and validation performance for plot in task.py
    5. save the model
    6. return None
    '''
    # intilse patience count for early stopping, and the list for storing train and validation accuracy and loss
    # set the label smoothing settings and min_delta (thershold) for early stopping
    save_train_acc = []
    save_train_loss = []
    save_val_acc = []
    save_val_loss = []
    early_stopping_patience = 0
    best_val_loss = np.inf

    min_delta = 0.01

    def weight_init(layer_in):
        '''
        Function for He initialisation
        '''
        if isinstance(layer_in, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(layer_in.weight, nonlinearity='relu')
            if layer_in.bias is not None:
                layer_in.bias.data.fill_(0.0)

    # Set model, He initialisation and device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModelClass.to(device)
    model.device = device
    model.apply(weight_init)

    # Set loss and optimiser, with learning rate scheduler 
    # lr_scheduler: decrease the learning rate a later of the traning loop
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)


    # train loop
    n_epochs = 50
    for epoch in range(n_epochs):

        # intilze the list for storing train_loss, train_accs for each epoch
        train_loss = []
        train_accs = []
        train_number = []
        # set model to train mode
        model.train()

        # train over each batch
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # make label become one-hot vector
            labels = F.one_hot(labels, num_classes = 10)
            # use mixup img and label for trianing
            inputs, labels  = MixUp(inputs, labels)
            
            # label smoothing
            labels = smoothing(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            
            # save the train_loss and train_accs
            acc = (outputs.argmax(dim=-1) == labels.argmax(dim =-1).to(device)).sum().item()
            
            train_number.append(len(labels))
            train_accs.append(acc)
            train_loss.append(loss.item())
        
        # update the step size and report mean train loss and accuracy for each epoch
        scheduler.step()
        mean_train_loss = sum(train_loss)/len(train_loss)
        total_train_accs = sum(train_accs)/sum(train_number)
        print(f"[ Train | {epoch + 1:03d} /{n_epochs:03d}] loss = {mean_train_loss:.5f}, acc = {total_train_accs:.5f}")


        
        # intilze the list for storing val_loss, val_accs for each epoch
        valid_loss = []
        valid_accs = []
        valid_number = []
        # set the model to eval mode
        model.eval()

        # validation
        for i, data in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # model evaluation no need to apply one-hot vector, MixUp and label smoothing
            inputs, labels = data


            # get and save the validation loss and accuracy
            with torch.no_grad():
                outputs = model(inputs.to(device))
            
            loss = criterion(outputs, labels.to(device))


            acc = (outputs.argmax(dim=-1) == labels.to(device)).sum().item()
            valid_number.append(len(labels))
            valid_accs.append(acc)
            valid_loss.append(loss.item())
        
        # report mean validation loss and accuracy for each epoch
        mean_val_loss = sum(valid_loss)/len(valid_loss)
        total_val_accs = sum(valid_accs)/sum(valid_number)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {mean_val_loss:.5f}, acc = {total_val_accs:.5f}")

        # save the training and validation performance
        save_train_acc.append(total_train_accs)
        save_train_loss.append(mean_train_loss)
        save_val_acc.append(total_val_accs)
        save_val_loss.append(mean_val_loss)
        
        # Early Stopping, with patience = 5
        if early_stopping_patience < 5:
            # only remember model when loss improved fulfill thershold
            # otherwise, consider as no improvment in this epoch and count patience
            if mean_val_loss  <  (best_val_loss - min_delta) :
                best_val_loss = mean_val_loss
                early_stopping_patience = 0
                best_model = model.state_dict()
            else: early_stopping_patience += 1
        else:
            break

        # report Early Stopping status
        print(f"[ patience: {early_stopping_patience} ] Best loss = {best_val_loss:.5f}, threashold = {(best_val_loss - min_delta):.5f}")


        


    print('Training done.')

    # save preformance to txt file

    with open("result.txt", 'a') as file:
        result = ','.join([str(i) for i in save_train_acc])
        file.write(result+ '\n')
        result = ','.join([str(i) for i in save_train_loss])
        file.write(result+ '\n')
        result = ','.join([str(i) for i in save_val_acc])
        file.write(result+ '\n')
        result = ','.join([str(i) for i in save_val_loss])
        file.write(result+ '\n')


    # save trained model with best loss from early stopping
    torch.save(best_model, fname + '_model.pt')
    print('Model saved.')





if __name__ == '__main__':
    # dataset:cifar-10 

    # transform to tensor for model to understand
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # transform to tensor and  do image augmentation for the new augmentation set
    # Regularisation: image augmentation to increase train dataset with various of image data
    transform_a = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAdjustSharpness(2),
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64  #dataset total size: 50000 trainset size:35000, validset size:7500, testset size:7500
    
    data = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform)

    # random split data for train, validation and test set, set manual_seed to make reproducible random split
    generator = torch.Generator().manual_seed(42)
    trainset, validset, testset  = torch.utils.data.random_split(data, [0.7, 0.15, 0.15], generator=generator)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=2)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Add the augmentated image to the tarining set and train regularized model
    print("Training  regularized model...")
    data = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform_a)
    Augment_train_dataset, _, _ = torch.utils.data.random_split(data, [0.7, 0.15, 0.15], generator=generator)
    train_dataset = torch.utils.data.ConcatDataset([trainset, Augment_train_dataset])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    train(Reg_CNN(), trainloader = trainloader, valloader = valloader, weight_decay = 3e-5, fname ="Regularized" )


