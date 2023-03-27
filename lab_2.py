import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
data = np.load('lab2_dataset.npz')
train_feats = torch.tensor(data['train_feats'])
train_feats = train_feats.to(device)
test_feats = torch.tensor(data['test_feats'])
test_feats = test_feats.to(device)
train_labels = torch.tensor(data['train_labels'])
train_labels = train_labels.to(device)
test_labels = torch.tensor(data['test_labels'])
test_labels = test_labels.to(device)
phone_labels = data['phone_labels']


# Set up the dataloaders
train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(test_feats, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)


# Define the model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Linear layers
        self.layer1 = nn.Linear(440, 1536)
        self.layer2 = nn.Linear(1536, 768)
        self.layer3 = nn.Linear(768, 384)
        self.layer4 = nn.Linear(384, 192)
        self.layer5 = nn.Linear(192, 96)
        self.layer6 = nn.Linear(96, 48)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Flatten the input
        x = torch.reshape(x, (-1, 11 * 40))
        count = 0
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]
        # Loop through the layers
        for i in self.layers:
            count += 1
            x = i(x)
            x = self.relu(x)
        return x
    
# Instantiate the model, loss function, and optimizer
model = MyModel()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_network(model, train_loader, criterion, optimizer):
    for epoch in range(10):
        start = time.time()
        # Training loop
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        end = time.time()
        print('Epoch %d' % (epoch + 1))
        print(end - start)
        # Testing loop
        test_network(model, test_loader)


def test_network(model, test_loader):
    correct = 0
    total = 0
    # accuracy for each class
    correct_count = [0 for i in range(len(phone_labels))]
    # total number of each class
    total_count = [0 for i in range(len(phone_labels))]
    # miss classification matrix
    miss = [[0 for x in range(len(phone_labels))] for y in range(len(phone_labels))]
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            # outputs 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(predicted)):
                # increment total count for each class
                total_count[predicted[i]] += 1
                if predicted[i] == labels[i]:
                    # increment correct count for each class
                    correct_count[predicted[i]] += 1
                else:
                    # increment miss classification matrix
                    miss[labels[i]][predicted[i]] += 1

    print('Test accuracy: %d %%' % (100 * correct / total))
    return np.divide(correct_count, total_count), miss



# Putting it all together

def train_model():
    start = time.time()
    train_network(model, train_loader, criterion, optimizer)
    end = time.time()
    print(end - start)
    torch.save(model.state_dict(), 'model_1.pt')
def load_model():
    model.load_state_dict(torch.load('model_1.pt'))
    model.eval()

train_model()
# load_model()

# Testing the Model
accuracies, misses = test_network(model, test_loader)
label_accuracies = sorted(zip(accuracies, phone_labels), reverse=True)
print("3 highest:", label_accuracies[:3])
print("3 lowest:", label_accuracies[-3:])

for i in [14, 10, 39, 4, 35]:
    print(phone_labels[i], "is misclassified as", phone_labels[np.argmax(misses[i])], np.max(misses[i]), "times. Accuracy:", accuracies[i])
