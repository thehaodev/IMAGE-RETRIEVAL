import file_utils
import score_utils
import chromadb


def run_simple_test():
    root_img_path = f"{file_utils.ROOT}/train/"
    query_path = f"{file_utils.ROOT}/test/Orange_easy/0_100.jpg"
    size = (448, 448)

    # Run simple test with model
    score_type = score_utils.ScoreType.CORR_COEFF

    _, ls_path_score = score_utils.get_score_model(root_img_path, query_path, size, score_type=score_type)
    score_utils.plot_results(query_path, ls_path_score, top=5, column=3, score_type=score_type)


def run_test_model_clip():
    chroma_client = chromadb.Client()
    l2_collection = chroma_client.get_or_create_collection(name="l2_collection",
                                                           metadata={"hnsw:space": "l2"})
    score_utils.add_embedding(collection=l2_collection, files_path=file_utils.FILES_PATH)
    test_path = f'{file_utils.ROOT}/test'
    test_files_path = file_utils.get_files_path(path=test_path)
    test_path = test_files_path[1]
    l2_results = score_utils.search(image_path=test_path, collection=l2_collection, n_results=5)

    images_result = score_utils.get_image_from_search(file_utils.FILES_PATH, l2_results['ids'])
    score_utils.show_multi_image(top=5, column=3, img_path=images_result)


run_test_model_clip()
