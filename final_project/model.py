from utils import preprocess_data

import torch
from torch import nn
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, signals, labels):
        self.data = signals
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CNN_3axis(nn.Module):
    def __init__(self, num_class, hid_channel=196):
        super().__init__()
        self.num_class = num_class
        self.hid_channel = hid_channel
        self.cnn = nn.Conv2d(1, self.hid_channel, (1, 16))
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d((1, 4), stride=(1, 4))
        # self.maxpool = nn.MaxPool2d((1, 4))
        # self.fc = nn.Linear(self.hid_channel*3*18, 512)
        # self.dropout = nn.Dropout(0.5)        
        # self.fc = nn.Linear(self.hid_channel*3*18, 512)
        # self.dropout = nn.Dropout(0.5)
        # self.proj_class = nn.Linear(512, self.num_class)
        # self.cnn2 = nn.Conv2d(self.hid_channel, 2* self.hid_channel, (3, 3))
        self.proj = nn.Sequential(
            nn.Linear(self.hid_channel*3*65, 1024), 
            nn.ReLU(),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.relu(x)
        # x = self.cnn2(x)
        # x = self.relu(x)

        x = x.view(x.size()[0], -1)
        x = self.proj(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_class, out_channel=196):
        super().__init__()
        self.num_class = num_class
        self.out_channel = out_channel
        self.cnn = nn.Conv2d(1, self.out_channel, (1, 16))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2, 4), stride=(2, 4))
        self.fc = nn.Linear(self.out_channel*3*16, 256)
        self.dropout = nn.Dropout(0.5)
        self.proj_class = nn.Linear(256, self.num_class)


    def forward(self, x):
        x = self.cnn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.proj_class(x)
        return x

class CNN_dwt(nn.Module):
    def __init__(self, num_class, out_channel=196):
        super().__init__()
        self.num_class = num_class
        self.out_channel = out_channel
        self.cnn = nn.Conv2d(1, self.out_channel, (1, 16))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2, 4), stride=(2, 4))
        self.fc = nn.Linear(self.out_channel*3*6, 256)
        self.dropout = nn.Dropout(0.5)
        self.proj_class = nn.Linear(256, self.num_class)


    def forward(self, x):
        x = self.cnn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.proj_class(x)
        return x

def model():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")


    model = CNN_3axis(9, 128).to(device)
    print(model)

    # loss_fn = nn.CrossEntropyLoss()
    return model