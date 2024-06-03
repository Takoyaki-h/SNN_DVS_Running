import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerPolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_size=128, num_layers=8, num_thresholds=11):
        super(LayerPolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_layers * num_thresholds)
        self.num_layers = num_layers
        self.num_thresholds = num_thresholds

    def forward(self, x):
        bs, c, w, h = x.size()  # 获取输入张量的形状
        x = x.view(bs, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # 调整输出形状为 [bs, num_layers, num_thresholds] 并应用 softmax
        x = x.view(-1, self.num_layers, self.num_thresholds)
        x = F.softmax(x, dim=2)
        return x
