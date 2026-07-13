#   CNN

import cv2
import numpy as np
import torch
from sudoku_model import SudokuCNN   
import RealTimeSudokuSolver_pytorch as solver

def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuCNN().to(device)


weights_path = "digitRecognition_pytorch_converted.pth"   # یا "digitRecognition_pytorch_converted.pth"
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print("model loaded.")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("error : camera")
    exit()

old_sudoku = None
print("exit : enter q")

while True:
    ret, frame = cap.read()
    if not ret:
        print("error : frame")
        break

    sudoku_frame, old_sudoku = solver.recognize_and_solve_sudoku(frame, model, old_sudoku, device)
    showImage(sudoku_frame, "Real Time Sudoku Solver (PyTorch)", 1066, 600)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()