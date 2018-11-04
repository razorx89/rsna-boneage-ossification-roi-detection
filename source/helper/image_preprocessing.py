import cv2
import numpy as np


def preprocess_image(image, chunk_size=64):
    height, width = image.shape
    #clahe = [cv2.createCLAHE(2.0 * f, (width // (chunk_size // f), height // (chunk_size // f))) for f in [1, 2, 4]]
    clahe = cv2.createCLAHE(2.0, (width // chunk_size, height // chunk_size))
    return np.stack([clahe.apply(image)] * 3, axis=2)