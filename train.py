import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import time
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="data directory", type=str, default="flowers")
parser.add_argument("--save_dir", help="save directory", type=str, default=".")
parser.add_argument("--arch", help="architecture", type=str, default="vgg16")
parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.001)
parser.add_argument("--hidden_units", help="hidden units", type=int, default=512)
parser.add_argument("--epochs", help="epochs", type=int, default=1)
parser.add_argument("--gpu", help="use gpu", default=True)
args = parser.parse_args()


class train_model():
    def __init__(self):
        self.data_dir = args.data_dir.rstrip("/")
        self.train_dir = self.data_dir + '/train'
        self.valid_dir = self.data_dir + '/valid'
        self.test_dir = self.data_dir + '/test'
        
        self.save_dir = args.save_dir.rstrip("/")
        self.arch = args.arch
        self.learning_rate = args.learning_rate
        self.hidden_units = args.hidden_units
        self.epochs = args.epochs
        self.gpu = args.gpu
        
        self.device = torch.device("cpu")

        if self.gpu == "True" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
    def data_transforms(self):
        print(self.data_dir)
        print(self.train_dir)
        print(self.arch)
        self.train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

        self.test_data_transforms = transforms.Compose([transforms.Resize(225),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])])

        self.validation_data_transforms = transforms.Compose([transforms.Resize(225),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])])

        self.train_image_datasets = datasets.ImageFolder(self.train_dir, transform=self.train_data_transforms)
        self.test_image_datasets = datasets.ImageFolder(self.test_dir, transform=self.test_data_transforms)
        self.validate_image_datasets = datasets.ImageFolder(self.valid_dir, transform=self.validation_data_transforms)

        self.train_dataloaders = torch.utils.data.DataLoader(self.train_image_datasets, batch_size=64, shuffle=True)
        self.test_dataloaders = torch.utils.data.DataLoader(self.test_image_datasets, batch_size=64, shuffle=True)
        self.validation_dataloaders = torch.utils.data.DataLoader(self.validate_image_datasets, batch_size=64, shuffle=True)
        
    def label_mapping(self):
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    
    def get_reset50_model(self):
        resnet_model = models.resnet50(pretrained=True)
        resnet_model.fc = nn.Sequential( nn.Linear(2048, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_units, 102),
            nn.LogSoftmax(dim = 1)
            )
                # Only train the classifier parameters, feature parameters are frozen
        
        return resnet_model
    
    def get_vgg13_model(self):
        vgg_model = models.vgg13(pretrained=True)
        vgg_model.classifier = nn.Sequential( 
                    nn.Linear(25088, self.hidden_units),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(int(self.hidden_units), int(self.hidden_units/2)),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(self.hidden_units/2, 102),
                    nn.LogSoftmax(dim = 1)
                    )
        vgg_model.model.classifier = vgg_model.classifier
        return vgg_model
    
    def get_vgg16_model(self):
        vgg_model = models.vgg16(pretrained=True)
        vgg_model.classifier = nn.Sequential( 
                    nn.Linear(25088, self.hidden_units),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(self.hidden_units, int(self.hidden_units/2)),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(int(self.hidden_units/2), 102),
                    nn.LogSoftmax(dim = 1)
                    )
        return vgg_model
    
    def get_alexnet_model(self):
        alexnet_model = models.alexnet(pretrained=True)
        alexnet_model.classifier = nn.Sequential( 
                    nn.Linear(9216, self.hidden_units),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(self.hidden_units, int(self.hidden_units/2)),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(int(self.hidden_units/2), 102),
                    nn.LogSoftmax(dim = 1)
                    )
        alexnet_model.classifier = alexnet_model.classifier
        return alexnet_model
            
    def build_model(self):
        
        if self.arch == 'vgg13':
            self.model = self.get_vgg13_model()
            self.optimizer = optim.Adam(self.model.classifier.parameters(), self.learning_rate)
            
        elif self.arch == 'vgg16':
            self.model = self.get_vgg16_model()
            self.optimizer = optim.Adam(self.model.classifier.parameters(), self.learning_rate)
            
        elif self.arch == 'resnet50':
            self.model = self.get_reset50_model()
            self.optimizer = optim.Adam(self.model.fc.parameters(), self.learning_rate)
        
        elif self.arch == 'alexnet':
            self.model = self.get_alexnet_model()
            self.optimizer = optim.Adam(self.model.classifier.parameters(), self.learning_rate)
        else:
            self.model = models.vgg16(pretrained=True)            
            
        for param in self.model.parameters():
            if self.arch == 'resnet50' or 'alexnet':
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Define our new classifier
#         self.classifier = nn.Sequential( nn.Linear(25088, 512),
#                                     nn.ReLU(),
#                                     nn.Dropout(p=0.2),
#                                     nn.Linear(512, 256),
#                                     nn.ReLU(),
#                                     nn.Dropout(p=0.2),
#                                     nn.Linear(256, 102),
#                                     nn.LogSoftmax(dim = 1)
#                                     )


        # replace current models classifier with this one
        # self.model.classifier = self.classifier

        self.criterion = nn.NLLLoss()

        # # Only train the classifier parameters, feature parameters are frozen
        # self.optimizer = optim.Adam(self.model.classifier.parameters(), self.learning_rate)

        self.model.to(self.device);
        
    def train_model(self):
        steps = 0
        running_loss = 0
        print_every = 5

        for epoch in range(self.epochs):
            for inputs, labels in self.train_dataloaders:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.validation_dataloaders:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            logps = self.model.forward(inputs)
                            batch_loss = self.criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{self.epochs}.. "
                            f"Train loss: {running_loss/print_every:.3f}.. "
                            f"Test loss: {test_loss/len(self.validation_dataloaders):.3f}.. "
                            f"Test accuracy: {accuracy/len(self.validation_dataloaders):.3f}")
                    running_loss = 0
                    self.model.train()
                    
    def save_checkpoint(self):
        self.model.class_to_idx = self.train_image_datasets.class_to_idx

        if self.arch == 'resnet50':
            checkpoint = {'arch': self.arch,
                'classifier': self.model.fc,
                'class_to_idx': self.train_image_datasets.class_to_idx,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'state_dict': self.model.state_dict()}
        else:
            checkpoint = {'arch': self.arch,
                            'classifier': self.model.classifier,
                            'class_to_idx': self.train_image_datasets.class_to_idx,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'state_dict': self.model.state_dict()}

        torch.save(checkpoint, self.save_dir + '/checkpoint.pth')
            
    def create_and_train(self):
        self.data_transforms()
        self.build_model()
        self.train_model()
        self.save_checkpoint()
        
model = train_model()
model.create_and_train()

# python train.py --epochs=2
        