from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import file_utils
import os
import cv2
from chromadb.utils.embedding_functions.open_clip_embedding_function import OpenCLIPEmbeddingFunction
from PIL import Image
from tqdm import tqdm


def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query ** 2))
    data_norm = np.sqrt(np.sum(data ** 2, axis=axis_batch_size))
    return np.sum(data * query, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)


def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.mean((data - query) ** 2, axis=axis_batch_size)


def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_size)


def correlation_coefficient(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_mean = query - np.mean(query)
    data_mean = data - np.mean(data, axis=axis_batch_size, keepdims=True)
    query_norm = np.sqrt(np.sum(query_mean ** 2))
    data_norm = np.sqrt(np.sum(data_mean ** 2, axis=axis_batch_size))

    return np.sum(data_mean * query_mean, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)


def compute_score_by_type(query, images_np, score_type):
    if score_type == ScoreType.L1_SCORE:
        rates = absolute_difference(query, images_np)
    elif score_type == ScoreType.L2_SCORE:
        rates = mean_square_difference(query, images_np)
    elif score_type == ScoreType.COSINE_SIMILARITY:
        rates = cosine_similarity(query, images_np)
    else:
        rates = correlation_coefficient(query, images_np)

    return rates


def get_score(root_img_path, query_path, size, score_type):
    query = file_utils.read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in file_utils.CLASS_NAME:
            # mang numpy nhieu anh, paths
            path = root_img_path + folder
            images_np, images_path = file_utils.folder_to_images(path, size)
            # compute rate base on score type
            rates = compute_score_by_type(query, images_np, score_type)
            ls_path_score.extend(list(zip(images_path, rates)))

    return query, ls_path_score


embedd_function = OpenCLIPEmbeddingFunction()


def get_single_image_embedding(image):
    embedding = embedd_function._encode_image(image=image)
    return np.array(embedding)


def get_score_model(root_img_path, query_path, size, score_type):
    query = file_utils.read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in file_utils.CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = file_utils.folder_to_images(path, size)
            embedding_list = []

            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)

            rates = compute_score_by_type(query_embedding, np.stack(embedding_list), score_type)
            ls_path_score.extend(list(zip(images_path, rates)))

    return query, ls_path_score


def plot_results(query_path, ls_path_score: list, top, column, score_type):
    # Sort to get max
    ls_path_score.sort(key=lambda x: x[1])

    if (score_type == ScoreType.COSINE_SIMILARITY) or (score_type == ScoreType.CORR_COEFF):
        ls_path_score.reverse()

    img_path = [query_path]
    for i in range(top):
        img_path.append(ls_path_score[i][0])

    show_multi_image(top, column, img_path)


def add_embedding(collection, files_path):
    ids = []
    embeddings = []
    for id_filepath, filepath in tqdm(enumerate(files_path)):
        ids.append(f'id_{id_filepath}')
        image = Image.open(filepath)
        embedding = get_single_image_embedding(image=np.array(image.convert('L')))
        embeddings.append(embedding.tolist())

    collection.add(embeddings=embeddings, ids=ids)


def search(image_path, collection, n_results):
    query_image = Image.open(image_path)
    query_embedding = get_single_image_embedding(np.array(query_image.convert('L')))
    results = collection.query(query_embeddings=[query_embedding.tolist()],
                               n_results=n_results)
    return results


def get_image_from_search(files_path, search_id):
    list_image = []
    for id_filepath, filepath in enumerate(files_path):
        for id_img in search_id[0]:
            if f'id_{id_filepath}' == id_img:
                list_image.append(filepath)

    return list_image


def show_multi_image(top, column, img_path):
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


class ScoreType(Enum):
    L1_SCORE = 1
    L2_SCORE = 2
    COSINE_SIMILARITY = 3
    CORR_COEFF = 4
