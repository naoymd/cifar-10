import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import itertools
import matplotlib.pyplot as plt
from net import Net
from utils import imshowS

device = 'cuda' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_batch = 32
test_batch = 1
epoch_num = 50
class_num = 10

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net(class_num).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

l = []
net.train()
for epoch in range(epoch_num):
  class_correct = list(0. for i in range(class_num))
  class_total = list(0. for i in range(class_num))
  for i, data in enumerate(train_dataloader):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
      print('[epoch: %d, iter: %5d] loss: %.3f' % (epoch+1, i, loss))
      l.append(loss.item())

    predicted = torch.argmax(outputs, 1)
    boolean = (predicted == labels)
    correct = list(itertools.compress(predicted, boolean))
    for j in range(class_num):
      class_correct[j] += correct.count(j)
      class_total[j] += labels.tolist().count(j)
  
#   print(class_correct)
#   print(class_total)
  for i in range(class_num):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

net.eval()
net.load_state_dict(torch.load(PATH))
with torch.no_grad():
  class_correct = list(0. for i in range(class_num))
  class_total = list(0. for i in range(class_num))
  for i, data in enumerate(test_dataloader):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = net(inputs)

    predicted = torch.argmax(outputs, 1)
    boolean = (predicted == labels)
    correct = list(itertools.compress(predicted, boolean))

    if i % 100 == 0:
      imshow(torchvision.utils.make_grid(inputs.cpu()))
      print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(test_batch)))

    for j in range(class_num):
      class_correct[j] += correct.count(j)
      class_total[j] += labels.tolist().count(j)
  
  for i in range(class_num):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
  
plt.plot(l)
plt.show()