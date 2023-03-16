import torch
from torchvision import models
from torch import nn, optim
import numpy as np
import argparse
import json
from PIL import Image
from torchvision.transforms import functional as tvf

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="path to image", type=str, default="flowers")
parser.add_argument("--checkpoint", help="path to checkpoint", type=str, default="checkpoint.pth")
parser.add_argument("--category_names", help="category names", type=str, default="cat_to_name.json")
parser.add_argument("--topk", help="top number of items to return", type=int, default=1)
parser.add_argument("--gpu", help="use gpu", default=True)
parser.add_argument("--arch", help="architecture", type=str, default="vgg16")
args = parser.parse_args()


class predict():
    def __init__(self):
        self.image = args.image
        self.checkpoint = args.checkpoint
        self.category_names = args.category_names
        self.topk = args.topk
        self.gpu = args.gpu
        self.arch = args.arch
        
        self.device = torch.device("cpu")
        print(args.gpu)

        if self.gpu == "True" and torch.cuda.is_available():
            print("Using GPU")
            self.device = torch.device("cuda")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")
        
        self.cat_to_name = self.label_mapping()
        
    def label_mapping(self):
        with open(self.category_names, 'r') as f:
            return json.load(f)

    def load_checkpoint(self):
        print("loading checkpoint path: {}".format(self.checkpoint))
        checkpoint = torch.load(self.checkpoint)
        
        model = []
        if self.arch == 'vgg16':
            model = models.vgg16(pretrained=True)    
            model.classifier = checkpoint['classifier']
            optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.class_to_idx = checkpoint['class_to_idx']
            model.state_dict(checkpoint['state_dict'])   
        elif self.arch == 'vgg13':
            model = models.vgg13(pretrained=True)    
            model.classifier = checkpoint['classifier']
            optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.class_to_idx = checkpoint['class_to_idx']
            model.state_dict(checkpoint['state_dict'])   
        elif self.arch == 'alexnet':
            model = models.alexnet(pretrained=True)    
            model.classifier = checkpoint['classifier']
            optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.class_to_idx = checkpoint['class_to_idx']
            model.state_dict(checkpoint['state_dict'])               
        if self.arch == 'resnet50':
            model = models.resnet50(pretrained=True)    
            model.fc = checkpoint['classifier']
            optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.class_to_idx = checkpoint['class_to_idx']
            model.state_dict(checkpoint['state_dict'])
        
        return model
    
    def process_image(self):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        print("Processing image...")
        im = Image.open(self.image)
        im.thumbnail([256, 256],Image.ANTIALIAS)
        
        # using torchvision center_crop so there's no need to calculate the crop for each side
        im_cropped = tvf.center_crop(im, 224)
        
        np_image = np.array(im_cropped)
        np_image = np_image / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        np_image = (np_image - mean) / std
        processed_image = np.transpose(np_image, (2, 0, 1))
        return processed_image

    def predict_outcome(self):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''    
        print("Calcualting classification...")
        image_model = self.load_checkpoint()
        image_model.to(self.device);
        
        image = self.process_image()
        
        #convert to tensor from numpy
        image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
        
        image_tensor = torch.unsqueeze(image_tensor, 0) #Due to batch size
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = image_model.forward(image_tensor)
            topk_probs, topk_classes = torch.topk(output, self.topk) 
            topk_probs = torch.exp(topk_probs)
        
        # create an empty array to store the class labels in
        classes = []
        
        idx_to_class = {}
        for image_class in image_model.class_to_idx:
            value = image_model.class_to_idx[image_class]
            idx_to_class[value] = image_class
        
        # loop though predictions and populate classes
        for label in topk_classes.cpu().numpy()[0]:
            classes.append(idx_to_class[label])

        return topk_probs, classes
    
    def output(self):
        probs, classes = self.predict_outcome()
        
        probs = probs.data.cpu().numpy()[0]
        #class_names = classes.data.cpu().numpy()[0]
        
        labels = []
        
        for image_class in classes:
            labels.append(self.cat_to_name[image_class])
        
        print("Classification: {} \t Probability: {}".format(labels[0], probs[0]))
        
        if self.topk > 1:
            for i in range(len(labels)):
                print("Name: {} Class:{} Probaility: {}".format(labels[i], classes[i], probs[i]))
            
    def result(self):
        self.output()
        
prediction_model = predict()
prediction_model.result()

# python predict.py --image="flowers/test/1/image_06743.jpg"
        