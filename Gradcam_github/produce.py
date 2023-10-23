import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image

from Gradcam_github.utils import visualize_cam, Normalize
from Gradcam_github.gradcam import GradCAM, GradCAMpp

def get_mask(image,class_index = None):
    img_dir = 'images'
    # img_name = 'collies.JPG'
    # img_name = 'multiple_dogs.jpg'
    # img_name = 'snake.JPEG'
    img_name = image
    img_path = os.path.join(img_dir, img_name)
    pil_img = PIL.Image.open(img_path)
    pil_img = pil_img.convert("RGB")
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
    b,c,h,w = torch_img.shape
    torch_img = F.upsample(torch_img, size=(h, w), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)
    vgg = models.vgg16(pretrained=True)
    vgg.eval()
    cam_dict = dict()
    vgg_model_dict = dict(type='vgg', arch=vgg, layer_name='features_29', input_size=(h, w))
    vgg_gradcam = GradCAM(vgg_model_dict, True)
    vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
    cam_dict['vgg'] = [vgg_gradcam, vgg_gradcampp]
    mask, _ = vgg_gradcam(normed_torch_img, class_index)
    return mask,torch_img

def grad_cam(mask,torch_img):
    images = []
    heatmap, result = visualize_cam(mask, torch_img)
    # images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, result], 0))
    # images = make_grid(torch.cat(images, 0), nrow=5)
    # output_dir = 'outputs'
    # os.makedirs(output_dir, exist_ok=True)
    # output_name = "result.png"
    # output_path = os.path.join(output_dir, output_name)
    # save_image(images, output_path)
    # PIL.Image.open(output_path)
    return result