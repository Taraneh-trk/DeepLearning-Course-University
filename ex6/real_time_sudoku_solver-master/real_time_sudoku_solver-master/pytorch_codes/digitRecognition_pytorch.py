import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import random
import cv2
from scipy import ndimage
from sudoku_model import SudokuCNN  

def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0 - cx).astype(int)
    shifty = np.round(rows/2.0 - cy).astype(int)
    return shiftx, shifty

def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted

def shift_according_to_center_of_mass(img):
    img = cv2.bitwise_not(img)
    shiftx, shifty = get_best_shift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted
    img = cv2.bitwise_not(img)
    return img

batch_size = 128
num_classes = 9
epochs = 35
img_rows, img_cols = 28, 28

DATADIR = "DigitImages"
CATEGORIES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

if __name__ == "__main__":
    training_data = []
    def create_training_data():
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_rows, img_cols))
                new_array = shift_according_to_center_of_mass(new_array)
                training_data.append([new_array, class_num])

    create_training_data()
    random.shuffle(training_data)

    split = int(len(training_data) * 0.8)
    x_train = np.array([item[0] for item in training_data[:split]])
    y_train = np.array([item[1] for item in training_data[:split]])
    x_test  = np.array([item[0] for item in training_data[split:]])
    y_test  = np.array([item[1] for item in training_data[split:]])

    x_train = x_train.reshape(-1, 1, img_rows, img_cols).astype('float32') / 255.0
    x_test  = x_test.reshape(-1, 1, img_rows, img_cols).astype('float32') / 255.0

    x_train_t = torch.tensor(x_train)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_test_t  = torch.tensor(x_test)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train_t, y_train_t)
    test_dataset  = TensorDataset(x_test_t, y_test_t)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size)

    model = SudokuCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        acc = 100.0 * correct / total
        print(f'Epoch {epoch}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Test Acc: {acc:.2f}%')

    torch.save(model.state_dict(), 'digitRecognition_pytorch.pth')
    print("Model weights saved to digitRecognition_pytorch.pth")