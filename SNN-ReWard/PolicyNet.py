import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        bs, c, w, h = x.size()  # 获取输入张量的形状
        x = x.view(bs, -1)  # 将输入张量转换为[bs, c*w*h]的形状
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        prob = self.softmax(x)
        return prob

# # 初始化策略网络
# bs, c, w, h = 32, 3, 64, 64  # 示例输入形状
# input_size = c * w * h
# hidden_size = 128
# output_size = 11  # 阈值列表长度
# thresholds_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]  # 阈值列表
#
# p_net = PolicyNetwork(input_size, hidden_size, output_size)
#
# # 生成随机输入张量
# input_tensor = torch.randn(bs, c, w, h)
#
# # 获取每个阈值的选择概率
# probabilities = p_net(input_tensor)
# print("----probabilities.shape---")
# print(probabilities.shape)
#
# # 获取最大概率对应的阈值
# max_prob_index = torch.argmax(probabilities, dim=1)
# max_threshold = torch.tensor([thresholds_list[idx] for idx in max_prob_index])
# print('---max_threshold.shape----')
# print(max_threshold.shape)
#
# print("----max_threshold---")
# print(max_threshold)
