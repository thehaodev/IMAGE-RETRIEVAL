import os
from enum import Enum
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))


class ScoreType(Enum):
    L1_SCORE = 1
    L2_SCORE = 2
    COSINE_SIMILARITY = 3


def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)


def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query ** 2))
    data_norm = np.sqrt(np.sum(data ** 2, axis=axis_batch_size))
    return np.sum(data * query, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)


def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.mean((data - query) ** 2, axis=axis_batch_size)


def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)

    return images_np, images_path


def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_size)


def get_l1_score(root_img_path, query_path, size, score_type):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            # mang numpy nhieu anh, paths
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            # compute rate base on score type
            if score_type == ScoreType.L1_SCORE:
                rates = absolute_difference(query, images_np)
            elif score_type == ScoreType.L2_SCORE:
                rates = mean_square_difference(query, images_np)
            elif score_type == ScoreType.COSINE_SIMILARITY:
                rates = cosine_similarity(query, images_np)
            else:
                rates = absolute_difference(query, images_np)

            ls_path_score.extend(list(zip(images_path, rates)))

    return query, ls_path_score


def plot_results(query_path, ls_path_score: list, top, column, score_type):
    # Sort to get max
    ls_path_score.sort(key=lambda x: x[1])

    if score_type == ScoreType.COSINE_SIMILARITY:
        ls_path_score.reverse()

    img_path = [query_path]
    for i in range(top):
        img_path.append(ls_path_score[i][0])

    # set up figure
    fig = plt.figure()
    row = (top + 1) // column

    for index, path in enumerate(img_path):
        str_path = str(path)
        img = cv2.imread(str_path)
        img_title = str_path.split('/')

        fig.add_subplot(row, column, index + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(img_title[len(img_title) - 2])

        if index == top:
            break

    plt.show()


def run_simple_test():
    root_img_path = f"{ROOT}/train/"
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    size = (448, 448)
    score_type = ScoreType.L1_SCORE
    _, ls_path_score = get_l1_score(root_img_path, query_path, size, score_type=score_type)
    plot_results(query_path, ls_path_score, top=5, column=3, score_type=score_type)


run_simple_test()
