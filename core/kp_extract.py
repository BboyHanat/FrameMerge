import torch
import numpy as np
from core.model_test import D2Net
from core.utils import preprocess_image
from core.pyramid import process_multiscale


def init_model(model_path, device):
    model = D2Net(model_file=model_path, use_relu=True, device=device)
    model.eval()
    return model


def extract(model, image, device):
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    input_image = preprocess_image(image, preprocessing='caffe')
    input_tensor = torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32), device=device)
    with torch.no_grad():
        key_points, scores, descriptors = process_multiscale(input_tensor, model, scales=[1])

    key_points = key_points[:, [1, 0, 2]]
    feat = dict()
    feat['key_points'] = key_points
    feat['scores'] = scores
    feat['descriptors'] = descriptors

    return feat
