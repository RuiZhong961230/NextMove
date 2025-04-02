import torch.nn as nn

# 4. Next Location Prediction

class MyFullyConnect(nn.Module):
    #  __init__是构造函数，用于初始化网络结构
    def __init__(self, input_dim, output_dim):     
        super(MyFullyConnect, self).__init__()

        self.block = nn.Sequential(                 # 这是一个 nn.Sequential 容器，它包含了一系列的层，这些层将按顺序应用于输入数据。
            nn.Linear(input_dim, input_dim*2),      # 第一个全连接层 (32, 64)
            nn.ReLU(),                              # 激活函数，用于引入非线性。
            nn.Dropout(0.1),                        # 正则化技术，随机丢弃10%的节点特征，以防止过拟合。
            nn.Linear(input_dim*2, input_dim),      # 第二个全连接层 (64, 32)
            nn.Dropout(0.1),                        # 正则化技术，随机丢弃10%的节点特征，以防止过拟合。
        )

        self.batch_norm = nn.BatchNorm1d(input_dim) # 批量归一化层，用于规范化输入数据，提高训练稳定性。
        self.drop = nn.Dropout(0.1)

        num_locations = output_dim
        self.linear_class1 = nn.Linear(input_dim, num_locations)    # 最终的第三个全连接层 (32, 2418 or 20607)

    # forward是前向传播函数，定义了数据通过网络的流程
    def forward(self, out):             
        x = out
        out = self.block(out)
        out = out + x
        out = self.batch_norm(out)
        out = self.drop(out)

        return self.linear_class1(out)
