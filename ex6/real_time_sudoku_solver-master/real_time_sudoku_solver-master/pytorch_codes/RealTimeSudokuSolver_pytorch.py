import cv2
import numpy as np
from scipy import ndimage
import math
import torch
import torch.nn as nn
import sudokuSolver          
import copy

class SudokuCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SudokuCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

def write_solution_on_image(image, grid, user_grid):
    SIZE = 9
    width = image.shape[1] // 9
    height = image.shape[0] // 9
    for i in range(SIZE):
        for j in range(SIZE):
            if user_grid[i][j] != 0:
                continue
            text = str(grid[i][j])
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseLine = cv2.getTextSize(text, font, fontScale=1, thickness=3)
            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width*j + math.floor((width - text_width) / 2) + off_set_x
            bottom_left_corner_y = height*(i+1) - math.floor((height - text_height) / 2) + off_set_y
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y),
                                font, font_scale, (0,255,0), thickness=3, lineType=cv2.LINE_AA)
    return image

def two_matrices_are_equal(matrix_1, matrix_2, row, col):
    for i in range(row):
        for j in range(col):
            if matrix_1[i][j] != matrix_2[i][j]:
                return False
    return True

def side_lengths_are_too_different(A, B, C, D, eps_scale):
    AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    AD = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
    CD = math.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)
    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest > eps_scale * shortest

def approx_90_degrees(angle, epsilon):
    return abs(angle - 90) < epsilon

def largest_connected_component(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    if len(sizes) <= 1:
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == max_label] = 0
    return img2

def angle_between(vector_1, vector_2):
    unit_vector_1 = vector_1 / (np.linalg.norm(vector_1) + 1e-10)
    unit_vector_2 = vector_2 / (np.linalg.norm(vector_2) + 1e-10)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    dot_product = np.clip(dot_product, -1.0, 1.0)   
    angle = np.arccos(dot_product)                 
    return angle * 57.2958

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

def get_corners_from_contours(contour, corner_amount=4, max_iter=200):
    coefficient = 1
    while max_iter > 0 and coefficient >= 0:
        max_iter -= 1
        epsilon = coefficient * cv2.arcLength(contour, True)
        poly_approx = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(poly_approx)
        if len(hull) == corner_amount:
            return hull
        else:
            if len(hull) > corner_amount:
                coefficient += 0.01
            else:
                coefficient -= 0.01
    return None

def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

def recognize_and_solve_sudoku(image, model, old_sudoku, device):
    clone_image = np.copy(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    biggest_contour = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            biggest_contour = c

    if biggest_contour is None:
        return image, old_sudoku

    corners = get_corners_from_contours(biggest_contour, 4)
    if corners is None:
        return image, old_sudoku

    rect = np.zeros((4, 2), dtype="float32")
    corners = corners.reshape(4, 2)

    # top left
    sum_val = 10000
    index = 0
    for i in range(4):
        if corners[i][0] + corners[i][1] < sum_val:
            sum_val = corners[i][0] + corners[i][1]
            index = i
    rect[0] = corners[index]
    corners = np.delete(corners, index, 0)

    # bottom right
    sum_val = 0
    for i in range(3):
        if corners[i][0] + corners[i][1] > sum_val:
            sum_val = corners[i][0] + corners[i][1]
            index = i
    rect[2] = corners[index]
    corners = np.delete(corners, index, 0)

    # top right / bottom left
    if corners[0][0] > corners[1][0]:
        rect[1] = corners[0]
        rect[3] = corners[1]
    else:
        rect[1] = corners[1]
        rect[3] = corners[0]

    rect = rect.reshape(4, 2)
    A, B, C, D = rect

    AB = B - A
    AD = D - A
    BC = C - B
    DC = C - D
    eps_angle = 20
    if not (approx_90_degrees(angle_between(AB, AD), eps_angle) and
            approx_90_degrees(angle_between(AB, BC), eps_angle) and
            approx_90_degrees(angle_between(BC, DC), eps_angle) and
            approx_90_degrees(angle_between(AD, DC), eps_angle)):
        return image, old_sudoku

    eps_scale = 1.2
    if side_lengths_are_too_different(A, B, C, D, eps_scale):
        return image, old_sudoku

    (tl, tr, br, bl) = rect
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    perspective_transformed_matrix = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, perspective_transformed_matrix, (max_width, max_height))
    orginal_warp = np.copy(warp)

    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    warp = cv2.GaussianBlur(warp, (5,5), 0)
    warp = cv2.adaptiveThreshold(warp, 255, 1, 1, 11, 2)
    warp = cv2.bitwise_not(warp)
    _, warp = cv2.threshold(warp, 150, 255, cv2.THRESH_BINARY)

    SIZE = 9
    grid = [[0]*SIZE for _ in range(SIZE)]

    height = warp.shape[0] // 9
    width = warp.shape[1] // 9
    offset_width = math.floor(width / 10)
    offset_height = math.floor(height / 10)

    for i in range(SIZE):
        for j in range(SIZE):
            crop_image = warp[height*i+offset_height:height*(i+1)-offset_height,
                              width*j+offset_width:width*(j+1)-offset_width]

            if crop_image.size == 0:
                continue

            ratio = 0.6
            while crop_image.shape[0] > 0 and np.sum(crop_image[0]) <= (1-ratio) * crop_image.shape[1] * 255:
                crop_image = crop_image[1:]
            while crop_image.shape[1] > 0 and np.sum(crop_image[:,-1]) <= (1-ratio) * crop_image.shape[1] * 255:
                crop_image = np.delete(crop_image, -1, 1)
            while crop_image.shape[0] > 0 and np.sum(crop_image[:,0]) <= (1-ratio) * crop_image.shape[0] * 255:
                crop_image = np.delete(crop_image, 0, 1)
            while crop_image.shape[1] > 0 and np.sum(crop_image[-1]) <= (1-ratio) * crop_image.shape[0] * 255:
                crop_image = crop_image[:-1]

            if crop_image.size == 0:
                continue

            crop_image = cv2.bitwise_not(crop_image)
            crop_image = largest_connected_component(crop_image)

            digit_pic_size = 28
            crop_image = cv2.resize(crop_image, (digit_pic_size, digit_pic_size))

            if crop_image.sum() >= digit_pic_size**2 * 255 - digit_pic_size * 1 * 255:
                continue

            center_width = crop_image.shape[1] // 2
            center_height = crop_image.shape[0] // 2
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            center_region = crop_image[x_start:x_end, y_start:y_end]
            if center_region.sum() >= center_width * center_height * 255 - 255:
                continue

            _, crop_image = cv2.threshold(crop_image, 200, 255, cv2.THRESH_BINARY)
            crop_image = crop_image.astype(np.uint8)

            crop_image = cv2.bitwise_not(crop_image)
            shift_x, shift_y = get_best_shift(crop_image)
            shifted = shift(crop_image, shift_x, shift_y)
            crop_image = shifted
            crop_image = cv2.bitwise_not(crop_image)

            crop_image = crop_image.astype('float32') / 255.0
            crop_tensor = torch.tensor(crop_image).unsqueeze(0).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                output = model(crop_tensor)
                pred = torch.argmax(output, dim=1).item() + 1
            grid[i][j] = pred

        # cv2.imshow(f'cell_{i}_{j}', crop_image)
        # cv2.waitKey(0)
        # print(f"Grid[{i}][{j}] = {grid[i][j]}")

    user_grid = copy.deepcopy(grid)

    if old_sudoku is not None and two_matrices_are_equal(old_sudoku, grid, 9, 9):
        if sudokuSolver.all_board_non_zero(grid):
            orginal_warp = write_solution_on_image(orginal_warp, old_sudoku, user_grid)
    else:
        sudokuSolver.solve_sudoku(grid)
        if sudokuSolver.all_board_non_zero(grid):
            orginal_warp = write_solution_on_image(orginal_warp, grid, user_grid)
            old_sudoku = copy.deepcopy(grid)

    result_sudoku = cv2.warpPerspective(orginal_warp, perspective_transformed_matrix,
                                        (image.shape[1], image.shape[0]),
                                        flags=cv2.WARP_INVERSE_MAP)
    result = np.where(result_sudoku.sum(axis=-1, keepdims=True) != 0, result_sudoku, image)
    return result, old_sudoku