# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021-2029 by
#    Mahmood Amintoosi <m.amintoosi@gmail.com>
#    All rights reserved.
#    MIT license.
"""Functions for converting a Fully Connected Layer to Fully Convolutional Layer in TF"""

# The main idea is borrowed from: https://learnopencv.com/cnn-fully-convolutional-image-classification-with-tensorflow/

import torch
import torch.nn as nn


class SeqModel(nn.Module):
    def __init__(self, inputSize):
        super(SeqModel, self).__init__()
        self.Conv1 = nn.Conv2d(3, 32, 3)
        self.Conv2 = nn.Conv2d(32, 64, 3)
        self.Conv3 = nn.Conv2d(64, 128, 3)
        self.Conv4 = nn.Conv2d(128, 128, 3)
        self.Conv5 = nn.Conv2d(128, 256, 3)
        self.Conv6 = nn.Conv2d(256, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # compute flatten size dynamically
        dummy = torch.zeros(1, 3, inputSize, inputSize)
        flat_size = self._forward_features(dummy).shape[1]
        self.Dense = nn.Linear(flat_size, 1)

    def _forward_features(self, x):
        x = self.pool(self.relu(self.Conv1(x)))
        x = self.pool(self.relu(self.Conv2(x)))
        x = self.pool(self.relu(self.Conv3(x)))
        x = self.pool(self.relu(self.Conv4(x)))
        x = self.relu(self.Conv5(x))
        x = self.pool(self.relu(self.Conv6(x)))
        return x.flatten(1)

    def forward(self, x):
        x = self._forward_features(x)
        return torch.sigmoid(self.Dense(x))


def seq_model(inputSize):
    return SeqModel(inputSize)


# setting FC weights to the final convolutional layer
def set_conv_weights(conv_model, feature_extractor):
    dense_layer = feature_extractor.Dense
    w = dense_layer.weight.data  # shape: (1, flat_size)
    b = dense_layer.bias.data    # shape: (1,)
    # reshape weight to (out_channels, in_channels, 1, 1)
    conv_model.lastConv.weight.data = w.view(1, -1, 1, 1)
    conv_model.lastConv.bias.data = b


class ConvModel(nn.Module):
    def __init__(self, inputSize):
        super(ConvModel, self).__init__()
        self.Conv1 = nn.Conv2d(3, 32, 3)
        self.MaxPool1 = nn.MaxPool2d(2, 2)
        self.Conv2 = nn.Conv2d(32, 64, 3)
        self.MaxPool2 = nn.MaxPool2d(2, 2)
        self.Conv3 = nn.Conv2d(64, 128, 3)
        self.MaxPool3 = nn.MaxPool2d(2, 2)
        self.Conv4 = nn.Conv2d(128, 128, 3)
        self.MaxPool4 = nn.MaxPool2d(2, 2)
        self.Conv5 = nn.Conv2d(128, 256, 3)
        self.Conv6 = nn.Conv2d(256, 512, 3)
        self.MaxPool5 = nn.MaxPool2d(2, 2)
        self.lastConv = nn.Conv2d(512, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.MaxPool1(self.relu(self.Conv1(x)))
        x = self.MaxPool2(self.relu(self.Conv2(x)))
        x = self.MaxPool3(self.relu(self.Conv3(x)))
        x = self.MaxPool4(self.relu(self.Conv4(x)))
        x = self.relu(self.Conv5(x))
        x = self.MaxPool5(self.relu(self.Conv6(x)))
        return torch.sigmoid(self.lastConv(x))


# Convert Model
def convert_model(model, mdl, inputSize):
    """
    Convert FC2FC

    Parameters
    ----------
    model : input model
    mdl: model
    inputSize: the size of input image 

    Returns
    -------
    converted model

    """
    convModel = ConvModel(inputSize)
    state_dict = torch.load(mdl, map_location='cpu')
    convModel.load_state_dict(state_dict, strict=False)
    set_conv_weights(convModel, model)
    return convModel


# Model definition
class SeqModelHoda(nn.Module):
    def __init__(self, input_size=26):
        super(SeqModelHoda, self).__init__()
        
        # Convolutional layers with NO padding (matching TensorFlow default)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)  # in_ch=1 → out_ch=32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)  # in_ch=32 → out_ch=64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)  # in_ch=64 → out_ch=128

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Output layer: input size after pooling → 1×1 × 128 → final vector length = 128
        self.dense1 = nn.Linear(128, 10)  # ✅ Correct input size

    def forward(self, x):
        # Step-by-step with no padding:
        
        # Input: (batch_size, 1, 26, 26)
        x = self.pool1(torch.relu(self.conv1(x)))   # After conv1 → (24×24), pool to (12×12)
        x = self.pool2(torch.relu(self.conv2(x)))   # After conv2 → (10×10), pool to (5×5)
        x = self.pool3(torch.relu(self.conv3(x)))   # After conv3 → (3×3), pool to (1×1)

        x = self.flatten(x)                        # Now: [batch, 128]
        x = torch.softmax(self.dense1(x), dim=1)  # Output: [batch, 10]
        return x


def seq_model_hoda(inputSize):
    return SeqModelHoda(inputSize)


# class ConvModelHoda(nn.Module):
#     def __init__(self):
#         super(ConvModelHoda, self).__init__()
#         self.Conv1 = nn.Conv2d(1, 32, 3)
#         self.MaxPool1 = nn.MaxPool2d(2, 2)
#         self.Conv2 = nn.Conv2d(32, 64, 3)
#         self.MaxPool2 = nn.MaxPool2d(2, 2)
#         self.Conv3 = nn.Conv2d(64, 128, 3)
#         self.MaxPool3 = nn.MaxPool2d(2, 2)
#         self.lastConv = nn.Conv2d(128, 10, 1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.MaxPool1(self.relu(self.Conv1(x)))
#         x = self.MaxPool2(self.relu(self.Conv2(x)))
#         x = self.MaxPool3(self.relu(self.Conv3(x)))
#         return torch.sigmoid(self.lastConv(x))

class FCNModelHoda(nn.Module):
    """تبدیل مدل SeqModelHoda به Fully Convolutional"""
    def __init__(self, trained_model):
        super(FCNModelHoda, self).__init__()
        
        self.conv1 = trained_model.conv1
        self.pool1 = trained_model.pool1
        self.conv2 = trained_model.conv2
        self.pool2 = trained_model.pool2
        self.conv3 = trained_model.conv3
        self.pool3 = trained_model.pool3
        
        dense_weight = trained_model.dense1.weight  # (10, 128)
        dense_bias = trained_model.dense1.bias      # (10,)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 26, 26)
            x = self.pool1(torch.relu(self.conv1(dummy)))
            x = self.pool2(torch.relu(self.conv2(x)))
            x = self.pool3(torch.relu(self.conv3(x)))
            _, C, H, W = x.shape
        
        self.lastConv = nn.Conv2d(C, 10, kernel_size=(H, W), bias=True)
        
        with torch.no_grad():
            # reshape: (10, 128) -> (10, C, H, W)
            self.lastConv.weight.data = dense_weight.view(10, C, H, W)
            self.lastConv.bias.data = dense_bias
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.lastConv(x)  # خروجی: (B, 10, H', W')
        return x

# Convert Model
def convert_model_hoda(model, mdl, inputSize):
    """
    Convert FC2FC for Hoda dataset

    Parameters
    ----------
    model : input model
    mdl: model
    inputSize: the size of input image 

    Returns
    -------
    converted model

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = SeqModelHoda()
    trained_model.load_state_dict(torch.load('models/best_model-CNN.pth', map_location=device))
    trained_model.eval()

    fcn_model = FCNModelHoda(trained_model).to(device)
    fcn_model.eval()

    print(fcn_model)
