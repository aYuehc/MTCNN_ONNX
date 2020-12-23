"""
================================================================
    Copyright (C) 2020 * Ltd. All rights reserved.
   
    Author      : AYueh
    Time        : 22:56, 2020/12/10
    Editor      : PyCharm
    File name   : get_nets.py
    Description :

================================================================
"""

import onnxruntime

def PNet(path="weights/pnet.onnx"):
    return onnxruntime.InferenceSession(path)

def RNet(path="weights/rnet.onnx"):
    return onnxruntime.InferenceSession(path)

def ONet(path="weights/onet.onnx"):
    return onnxruntime.InferenceSession(path)