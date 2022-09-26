from ImgNet import *
from time import sleep

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