from ImgNet import *
from time import sleep

test_set = datasets.CIFAR10(root="./data", train=False, transform=transformations, download=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
print("The number of images in a test set is: ", len(test_loader)*batch_size)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = Network()
path = "model/ImgNet.pth"
model.load_state_dict(torch.load(path))

def run():
    stream = iter(test_loader)
    for batch in stream:
        images, labels = batch
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # calculate accuracy predicted vs labels
        correct = len(list(filter(lambda x: x[0] == x[1], zip(predicted, labels))))
        print(f"Accuracy: {correct / batch_size}\n")
        sleep(1)
run()