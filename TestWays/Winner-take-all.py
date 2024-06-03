import torch

# 使用winner-take-all方法生成mask
def reparameterization_winner_take_all(input_tensor, beta):
    # 计算保留的特征数 K
    P = input_tensor.size(1) * input_tensor.size(2) * input_tensor.size(3)  # 输入特征的总数量
    K = int(beta * P)  # 根据屏蔽比例计算保留的特征数

    # 将输入特征展平为一维向量，并计算最大的前 K 个值及其对应的索引
    flattened_input = input_tensor.view(input_tensor.size(0), -1)
    topk_values, topk_indices = flattened_input.topk(K, dim=1)

    # 创建输出的掩码向量，保留最大的前 K 个值对应的位置
    mask = torch.zeros_like(flattened_input)
    mask.scatter_(1, topk_indices, 1)

    # 将掩码向量恢复成输入特征的形状
    mask = mask.view(input_tensor.size())
    print(input_tensor.shape)
    print(mask.shape)
    print("**************************")
    print(mask)

    # 应用掩码向量，屏蔽未被保留的特征
    masked_output = input_tensor * mask

    return masked_output


# 示例输入数据，假设有一个 batch size 为 2，通道数为 3，宽度为 4，高度为 4 的特征图
input_tensor = torch.rand(2, 3, 4, 4)
beta = 0.5  # 屏蔽比例为 0.5

# 调用 reparameterization_winner_take_all 函数
masked_output = reparameterization_winner_take_all(input_tensor, beta)
print(masked_output.size())  # 输出形状为 [2, 3, 4, 4]
