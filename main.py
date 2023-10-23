import numpy as np
from matplotlib import pyplot as plt
import cv2
from draw import feature_map_class
from Gradcam_github.produce import grad_cam
from Gradcam_github.produce import get_mask
from seam_carving import find_low_energy_seam, remove_seam, mark_red
from PIL import Image
import vectorize


def main():
    ROUND = 101
    (mask, image) = get_mask("mountain_boat.png")
    feature_map = feature_map_class(mask, image)
    feature_map.paint_featuremap()
    feature_map.mask = feature_map.mask.numpy()
    feature_map.image = feature_map.image.numpy()
    vectorized_image = np.zeros_like(feature_map.image)
    gradient = get_gradient(feature_map.image)
    heat_map = feature_map.mask * gradient
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    b, c, h, w = feature_map.image.shape
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (h, w))
    horiontal_seams = []
    vertical_seams = []
    h_flag = False
    for i in range(ROUND):
        # find seam
        # first horizontal carve then vertical
        seam_vertical,v_energy = find_low_energy_seam(heat_map.squeeze().squeeze())
        hori_heat_map = heat_map.transpose(0, 1, 3, 2)
        seam_horizontal,h_energy = find_low_energy_seam(hori_heat_map.squeeze().squeeze())
        if h_energy < v_energy:
            horiontal_seams.append(seam_horizontal)
            feature_map.image = feature_map.image.transpose(0, 1, 3, 2)
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
        cv2.waitKey(10)

        # remove seam
        feature_map.image, heat_map = remove_seam(seam, feature_map.image, heat_map)

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
            heat_map = heat_map.transpose(0, 1, 3, 2)
        h_flag = False
        # save progress
        if i % 100 == 0:
            # BGR to RGB
            image_to_save = Image.fromarray(image_bgr[:, :, ::-1])
            image_to_save.save(f'image_{i}.png')
        print(f"carving,progress {i} / {ROUND}...")
    out.release()
    cv2.destroyAllWindows()
    print("retracing carving...")
    x_space, y_space = vectorize.find_space_by_seams(horiontal_seams, vertical_seams, feature_map.image.squeeze())
    print("creating grid and triangles...")
    grid = vectorize.create_grid(feature_map.image.squeeze())
    triangles = vectorize.create_triangle(grid)
    # print("ploting grid and triangles")
    # vectorize.plot_grid_triangles(grid,triangles)
    print("moving triangles...")
    moved_triangles = vectorize.move_triangles_to_origin(x_space, y_space, triangles)
    # print("ploting grid and moved triangles")
    # vectorize.plot_grid_triangles(grid,moved_triangles)
    triangles = vectorize.create_triangle(grid)
    print("vectorizing image....")
    vectorize.vectorize_image(vectorized_image, moved_triangles, triangles, feature_map.image.squeeze())
    feature_map.image = vectorized_image
    image_to_save = Image.fromarray((feature_map.image[0].transpose(1, 2, 0) * 255).astype('uint8'))
    image_to_save.save(f'result.png')
    print("finished")


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
