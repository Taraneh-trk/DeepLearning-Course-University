import os
import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import RealTimeSudokuSolver

class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # index 0
            nn.ReLU(inplace=True),             # index 1
            nn.Conv2d(32, 64, kernel_size=3),  # index 2
            nn.ReLU(inplace=True),             # index 3
            nn.MaxPool2d(kernel_size=2),       # index 4
            nn.Dropout(0.25),                  # index 5
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),      # index 0
            nn.ReLU(inplace=True),             # index 1
            nn.Dropout(0.5),                   # index 2
            nn.Linear(128, 9),                 # index 3
        )

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

pth_file = 'digitRecognition_pytorch_converted.pth'
h5_file = 'digitRecognition.h5'

if not os.path.exists(pth_file):
    print("error : file note found.")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
    except ImportError:
        raise ImportError("error")

    keras_model = Sequential()
    keras_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    keras_model.add(Conv2D(64, (3, 3), activation='relu'))
    keras_model.add(MaxPooling2D(pool_size=(2, 2)))
    keras_model.add(Dropout(0.25))
    keras_model.add(Flatten())
    keras_model.add(Dense(128, activation='relu'))
    keras_model.add(Dropout(0.5))
    keras_model.add(Dense(9, activation='softmax'))
    keras_model.load_weights(h5_file)

    pytorch_model = DigitCNN()
    state_dict = pytorch_model.state_dict()

    w, b = keras_model.layers[0].get_weights()
    # Keras: (3,3,1,32) -> PyTorch: (32,1,3,3)
    state_dict['features.0.weight'] = torch.tensor(np.transpose(w, (3, 2, 0, 1)).copy())
    state_dict['features.0.bias']   = torch.tensor(b.copy())

    w, b = keras_model.layers[1].get_weights()
    # Keras: (3,3,32,64) -> PyTorch: (64,32,3,3)
    state_dict['features.2.weight'] = torch.tensor(np.transpose(w, (3, 2, 0, 1)).copy())
    state_dict['features.2.bias']   = torch.tensor(b.copy())

    w, b = keras_model.layers[5].get_weights()
    # Keras: (9216, 128) -> PyTorch: (128, 9216)
    state_dict['classifier.0.weight'] = torch.tensor(np.transpose(w).copy())
    state_dict['classifier.0.bias']   = torch.tensor(b.copy())

    w, b = keras_model.layers[7].get_weights()
    # Keras: (128, 9) -> PyTorch: (9, 128)
    state_dict['classifier.3.weight'] = torch.tensor(np.transpose(w).copy())
    state_dict['classifier.3.bias']   = torch.tensor(b.copy())

    pytorch_model.load_state_dict(state_dict)
    torch.save(pytorch_model.state_dict(), pth_file)
    print(f"file created sucssussfully {pth_file} .")

model = DigitCNN()
state_dict = torch.load(pth_file, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
