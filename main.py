import os

import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2
from draw import feature_map_class
from Gradcam_github.produce import grad_cam
from Gradcam_github.produce import get_mask
from seam_carving import find_low_energy_seam, remove_seam, mark_red
from PIL import Image
import vectorize
import argparse
from MiDaS import get_depth


def main():
    parser = argparse.ArgumentParser(
        description="Please specify the input image name and the number of seams to remove")
    parser.add_argument("--input_image", "-i", help="input image to process", default="mountain_boat.png")
    parser.add_argument("--seams", "-s", help="number of seams to carve", default=20)
    parser.add_argument("--grid", "-g", help="the option to plot grid, pay attention that this may take very long time "
                                             "to finish for large image file, default false", default=False)
    parser.add_argument("--mask_source", "-m",
                        help="choose the source of mask,vgg->gradcam by vgg, MiDaS->Depth map created by MiDaS,default gradcam",default= "vgg" )
    args = parser.parse_args()
    ROUND = int(args.seams)
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # the input image should be put in the "/images" folder
    (mask, image) = get_mask(args.input_image)
    if args.mask_source == 'MiDaS':
        mask = torch.from_numpy(get_depth(args.input_image)).unsqueeze(0).unsqueeze(0)
    feature_map = feature_map_class(mask, image)
    feature_map.paint_featuremap()
    feature_map.mask = feature_map.mask.numpy()
    feature_map.image = feature_map.image.numpy()
    vectorized_image = np.zeros_like(feature_map.image)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    b, c, h, w = feature_map.image.shape
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (h, w))
    horiontal_seams = []
    vertical_seams = []
    h_flag = False
    for i in range(ROUND + 1):
        gradient = get_gradient(feature_map.image)
        heat_map = feature_map.mask * gradient
        # find seam
        # first horizontal carve then vertical
        seam_vertical, v_energy = find_low_energy_seam(heat_map.squeeze().squeeze())
        hori_heat_map = heat_map.transpose(0, 1, 3, 2)
        seam_horizontal, h_energy = find_low_energy_seam(hori_heat_map.squeeze().squeeze())
        if h_energy < v_energy:
            horiontal_seams.append(seam_horizontal)
            feature_map.image = feature_map.image.transpose(0, 1, 3, 2)
            feature_map.mask = feature_map.mask.transpose(0, 1, 3, 2)
            heat_map = heat_map.transpose(0, 1, 3, 2)
            seam = seam_horizontal
            h_flag = True
        else:
            vertical_seams.append(seam_vertical)
            seam = seam_vertical
        # mark red and show it
        marked_image = mark_red(seam, feature_map.image)
        marked_image = marked_image[0].transpose(1, 2, 0)[:, :, ::-1] * 255
        marked_image = marked_image.astype('uint8')
        if h_flag:
            marked_image = marked_image.transpose(1, 0, 2)
        out.write(marked_image)
        cv2.imshow('Seam carving', marked_image)
        cv2.waitKey(1)

        # remove seam
        feature_map.image, feature_map.mask = remove_seam(seam, feature_map.image, feature_map.mask)

        # show the removed image
        image_bgr = feature_map.image[0].transpose(1, 2, 0)[:, :, ::-1] * 255
        image_bgr = image_bgr.astype('uint8')
        if h_flag:
            image_bgr = image_bgr.transpose(1, 0, 2)
        h, w, c = image_bgr.shape
        out.set(3, w)
        out.set(4, h)
        out.write(image_bgr)
        cv2.imshow('Seam carving', image_bgr)
        cv2.waitKey(1)

        if h_flag:
            feature_map.image = feature_map.image.transpose(0, 1, 3, 2)
            feature_map.mask = feature_map.mask.transpose(0, 1, 3, 2)
            heat_map = heat_map.transpose(0, 1, 3, 2)
        h_flag = False
        # save progress
        if i % 50 == 0:
            # BGR to RGB
            image_to_save = Image.fromarray(image_bgr[:, :, ::-1])
            image_to_save.save(os.path.join(output_folder, f'image_{i}.png'))
        print(f"carving,progress {i} / {ROUND}...")
    out.release()
    cv2.destroyAllWindows()
    print("retracing carving...")
    x_space, y_space = vectorize.find_space_by_seams(horiontal_seams, vertical_seams, feature_map.image.squeeze())
    print("creating grid and triangles...")
    grid = vectorize.create_grid(feature_map.image.squeeze())
    triangles = vectorize.create_triangle(grid)
    if args.grid:
        print("ploting grid and triangles")
        vectorize.plot_grid_triangles(grid, triangles, os.path.join(output_folder, f'original_grid.png'))
    print("moving triangles...")
    moved_triangles = vectorize.move_triangles_to_origin(x_space, y_space, triangles)
    if args.grid:
        print("ploting grid and moved triangles")
        vectorize.plot_grid_triangles(grid, moved_triangles, os.path.join(output_folder, f'moved_gird.png'))
    triangles = vectorize.create_triangle(grid)
    print("vectorizing image....")
    vectorize.vectorize_image(vectorized_image, moved_triangles, triangles, feature_map.image.squeeze())
    feature_map.image = vectorized_image
    image_to_save = Image.fromarray((feature_map.image[0].transpose(1, 2, 0) * 255).astype('uint8'))
    image_to_save.save(os.path.join(output_folder, f'result.png'))
    print("Finished! files saved to output folder")


def get_gradient(image):
    gradient_x = np.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    gradient_y = np.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    pad_x = image[:, :, :, 0:1]
    pad_y = image[:, :, 0:1, :]
    gradient_x = np.concatenate((pad_x, gradient_x), axis=3)
    gradient_y = np.concatenate((pad_y, gradient_y), axis=2)
    gradient = np.sum((gradient_y + gradient_x), axis=1, keepdims=True)
    return gradient


if __name__ == "__main__":
    main()
