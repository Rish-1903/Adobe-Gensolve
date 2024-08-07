import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
device='cuda' if torch.cuda.is_available() else 'cpu'
batch_size=200
# Define transformation for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images if necessary
    transforms.ToTensor()
])

# Load the dataset
train_dataset = datasets.ImageFolder(root='dataset/train/', transform=transform)
test_dataset=datasets.ImageFolder(root='dataset/valid',transform=transform)
# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1=nn.Conv2d(3,16,3,1,1)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(16,32,3,1,1)
        self.pool2=nn.MaxPool2d(2,2)
        self.conv3=nn.Conv2d(32,64,3,1,1)
        self.pool3=nn.MaxPool2d(1,1)
        self.conv4=nn.Conv2d(64,128,3,1,1)
        self.pool4=nn.AvgPool2d(1,1)
        self.conv5=nn.Conv2d(128,256,3,1,1)
        self.fc1=nn.Linear(256*32*32,120)
        self.fc2=nn.Linear(120,64)
        self.fc3=nn.Linear(64,32)
        self.fc4=nn.Linear(32,6)
    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=self.pool3(F.relu(self.conv3(x)))
        x=self.pool4(F.relu(self.conv4(x)))
        x=F.relu(self.conv5(x))
        x=x.view(-1,256*32*32)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x

learning_rate=0.001
model=ConvNet().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
n_total_steps=len(train_loader)
num_epochs=100
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%2000==0:
            print(f'Epoch [{epoch+1}/{num_epochs}],Step [{i+1}/{n_total_steps}],Loss:{loss.item():.4f}')

print('Finished Training')

def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

evaluate_model(model, test_loader, device)


def predict_image(image_path, model, transform, device):
    model.eval()  # Set the model to evaluation mode
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.item()  # Returns the index of the predicted class

# Example usage
image_path = "./dataset/valid/Star/star_320.png"
predicted_class = predict_image(image_path, model, transform, device)
class_names = train_dataset.classes
print(f'Predicted class name: {class_names[predicted_class]}')


        
        
        
        
        