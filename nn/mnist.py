import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
train, val = random_split(train_data, [55000,5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)

# define model
model = nn.Sequential(
    nn.Linear(28*28,64),
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64,10),
)

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# define loss 
loss = nn.CrossEntropyLoss()

# train loops
nb_epochs = 5
for epoch in range(nb_epochs):
    for batch in train_loader:
        x,y = batch # input and label

        b = x.size(0) # batch size
        x = x.view(b,-1) # image x: b x 1 x 28 x 28

        logits = model(x) # forward pass to get prob array
        L = loss(logits, y) # calc loss
        model.zero_grad() # clean grads

        L.backward() # backprop
        optimizer.step()

        print(f'Epoch {epoch+1}, train loss: {L.item()}')

# validation step
for batch in train_loader:
    x,y = batch # input and label

    b = x.size(0) # batch size
    x = x.view(b,-1) # image x: b x 1 x 28 x 28

    with torch.no_grad():
        logits = model(x) # forward pass to get prob array
    L = loss(logits, y) # calc loss


    print(f'Epoch {epoch+1}, val loss: {L.item()}')