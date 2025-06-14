import cv2
import numpy as np
from box import Box
from train import *
import os
import pickle

def predict(img: np.ndarray, model_path: str = 'trained_models/nn_trained_model_hog.sav', dataset_path: str = DEFAULT_DATASET_PATH) -> np.ndarray:
    """Predict symbol label for the given image using a saved model.

    If the model file does not exist it will be trained automatically using the
    dataset at ``dataset_path``. ``model_path`` should point to a pickle file
    produced by :func:`train.train`.
    """

    if not os.path.exists(model_path):
        print('Please wait while training the NN-HOG model....')
        base = os.path.splitext(os.path.basename(model_path))[0]
        train('NN', 'hog', base, dataset_path=dataset_path)

    model = pickle.load(open(model_path, 'rb'))
    features = extract_features(img, 'hog')
    labels = model.predict([features])

    return labels


# if __name__ == "__main__":
#     img = cv2.imread('testresult/0_6.png')
#     labels = predict(img)
#     print(labels)

