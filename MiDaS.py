import os

import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt


# This code is copied from pytorch documentation "https://pytorch.org/hub/intelisl_midas_v2/"
def get_depth(image_name):
    model_type = "MiDaS_small"  # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    img = cv2.imread(os.path.join("images", image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    normalized_output = (output - output.min()) / (output.max() - output.min())
    return normalized_output
# get_depth("mountain_boat.png")