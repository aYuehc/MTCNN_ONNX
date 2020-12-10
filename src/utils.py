"""
================================================================
    Copyright (C) 2020 * Ltd. All rights reserved.
   
    Author      : AYueh
    Time        : 23:17, 2020/12/10
    Editor      : PyCharm
    File name   : utils.py
    Description :

================================================================
"""


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()