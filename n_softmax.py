#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from tools import *


class Normalsoftmax(nn.Module):
    def __init__(self, nClasses):
        super(Normalsoftmax, self).__init__()

        self.test_normalize = True

        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = nn.Linear(192, nClasses)

        print('Initialised Softmax Loss')

    def forward(self, x, label=None):
        x = self.fc(x)
        nloss = self.criterion(x, label)
        prec1 = accuracy(x.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1