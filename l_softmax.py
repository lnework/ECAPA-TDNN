#导入库
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import math
import torch

class AngleLinear(nn.Module):#定义最后一层
    def __init__(self, in_features, out_features, m=3, phiflag=True):#输入特征维度，输出特征维度，margin超参数
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))#本层权重
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)#初始化权重，在第一维度上做normalize
        self.m = m
        self.phiflag = phiflag
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]#匿名函数,用于得到cos_m_theta

    @staticmethod
    def myphi(x, m):
        x = x * m
        return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) +\
               x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)

    def forward(self, x):#前向过程，输入x
        w = self.weight

        ww = w.renorm(2, 1, 1e-5).mul(1e5)#方向0上做normalize
        x_len = x.pow(2).sum(1).pow(0.5)
        w_len = ww.pow(2).sum(0).pow(0.5)

        cos_theta = x.mm(ww)
        cos_theta = cos_theta / x_len.view(-1, 1) / w_len.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1) #将数据夹在-1-1之间

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)#由m和/cos(/theta)得到cos_m_theta
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k#得到/phi(/theta)
        else:
            theta = cos_theta.acos()#acos得到/theta
            phi_theta = self.myphi(theta, self.m)#得到/phi(/theta)
            phi_theta = phi_theta.clamp(-1*self.m, 1)#控制在-m和1之间

        cos_theta = cos_theta * x_len.view(-1, 1)
        phi_theta = phi_theta * x_len.view(-1, 1)
        output = [cos_theta, phi_theta]#返回/cos(/theta)和/phi(/theta)
        return output

class AngleLoss(nn.Module):#设置loss，超参数gamma，最小比例，和最大比例
    def __init__(self, gamma=0, lambda_min=5, lambda_max=1500):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, x, y): #分别是output和target
        self.it += 1
        cos_theta, phi_theta = x #output包括上面的[cos_theta, phi_theta]
        y = y.view(-1, 1)

        index = cos_theta.data * 0.0
        index.scatter_(1, y.data.view(-1, 1), 1)#将label存成稀疏矩阵
        index = index.byte()
        index = Variable(index)

        lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.it))#动态调整lambda，来调整cos(\theta)和\phi(\theta)的比例
        output = cos_theta * 1.0
        output[index] -= cos_theta[index]*(1.0+0)/(1 + lamb)#减去目标\cos(\theta)的部分
        output[index] += phi_theta[index]*(1.0+0)/(1 + lamb)#加上目标\phi(\theta)的部分

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, y)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss, 0