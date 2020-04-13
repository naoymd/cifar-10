import torch
import torch.nn
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self, class_num):
        super(Net, self).__init__()
        self.resnet = models.resnet50()
        self.fc = nn.LInear(1000, class_num)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
