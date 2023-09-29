#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 13:01:53 2023

@author: sascha
"""

import torch

J = 8
y = torch.tensor([28, 8, -3, 7, -1, 1, 18, 12]).type(torch.Tensor)
sigma = torch.tensor([15, 10, 16, 11, 9, 11, 10, 18]).type(torch.Tensor)