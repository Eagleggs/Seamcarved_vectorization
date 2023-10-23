import numpy as np

# creat a grid by using the center fo the pixel as the vertices
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon


def create_grid(image):
    c, h, w = image.shape
    x = np.arange(0.5, w - 1e-3, 1)
    y = np.arange(0.5, h - 1e-3, 1)
    X, Y = np.meshgrid(x, y)
    coordinates = np.stack((X, Y), axis=-1)
    return coordinates


# create triangles, with the orientation |/| then |\|

def create_triangle(grid):
    h, w, c = grid.shape
    triangle = np.zeros((h - 1, 2 * (w - 1), 3, 2))
    for y in range(h - 1):
        for x in range(0, 2 * (w - 1), 2):
            i = int(x / 2)
            if y % 2 == 0:
                if i % 2 == 0:
                    triangle[y, x] = [grid[y, i], grid[y, i + 1], grid[y + 1, i]]
                    triangle[y, x + 1] = [grid[y, i + 1], grid[y + 1, i + 1], grid[y + 1, i]]
                else:
                    triangle[y, x] = [grid[y, i], grid[y + 1, i + 1], grid[y + 1, i]]
                    triangle[y, x + 1] = [grid[y, i], grid[y, i + 1], grid[y + 1, i + 1]]
            else:
                if i % 2 == 0:
                    triangle[y, x] = [grid[y, i], grid[y + 1, i + 1], grid[y + 1, i]]
                    triangle[y, x + 1] = [grid[y, i], grid[y, i + 1], grid[y + 1, i + 1]]
                else:
                    triangle[y, x] = [grid[y, i], grid[y, i + 1], grid[y + 1, i]]
                    triangle[y, x + 1] = [grid[y, i + 1], grid[y + 1, i + 1], grid[y + 1, i]]
    return triangle


def barycentric_coordinates(triangle, point):
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = point - triangle[0]
    d00 = np.sum(v0 * v0)
    d01 = np.sum(v0 * v1)
    d11 = np.sum(v1 * v1)
    d21 = np.sum(v2 * v1)
    d20 = np.sum(v2 * v0)
    denom = d00 * d11 - d01 * d01
    y = (d11 * d20 - d01 * d21) / denom
    z = (d00 * d21 - d01 * d20) / denom
    x = 1 - y - z
    return [x, y, z]


def is_inside_triangle(barycentric_coor):
    # Check if the point is inside the triangle
    return 0 <= barycentric_coor[0] <= 1 and 0 <= barycentric_coor[1] <= 1 and 0 <= barycentric_coor[2] <= 1


def get_coordinate(barycentric_coor, triangle):
    assert (is_inside_triangle(barycentric_coor))
    x, y, z = barycentric_coor
    x0, y0 = triangle[0]
    x1, y1 = triangle[1]
    x2, y2 = triangle[2]

    # Calculate the actual coordinates of the point within the triangle
    x_point = x0 * x + x1 * y + x2 * z
    y_point = y0 * x + y1 * y + y2 * z

    return [x_point, y_point]


def sampleBilinear(image, pos_px):
    # Determine the 4 nearest pixels
    x_frac = pos_px[0] - int(pos_px[0])
    y_frac = pos_px[1] - int(pos_px[1])
    x0 = int(pos_px[0]) if x_frac - 0.5 > 0 else int(pos_px[0]) - 1
    y0 = int(pos_px[1]) if y_frac - 0.5 > 0 else int(pos_px[1]) - 1
    width, height = image.shape[2], image.shape[1]
    if (x0 >= width):
        x0 = width - 1
    if (y0 >= height):
        y0 = height - 1

    x1 = x0 + 1
    y1 = y0 + 1
    if (x1 < 0):
        x1 = 0
    if (y1 < 0):
        y1 = 0
    alpha = pos_px[0] - x0 - 0.5
    beta = y1 - pos_px[1] + 0.5

    # Bilinear interpolation
    data_00 = image[:, y0, x0] if (x0 >= 0 and y0 >= 0) else (
        image[:, y1, x1] if x0 < 0 and y0 < 0 else (image[:, y0, x1] if x0 < 0 else image[:, y1, x0]))

    data_01 = image[:, y1, x0] if (x0 >= 0 and y1 < height) else (
        image[:, y0, x1] if x0 < 0 and y1 == height else (image[:, y1, x1] if x0 < 0 else image[:, y0, x0]))

    data_10 = image[:, y0, x1] if (y0 >= 0 and x1 < width) else (
        image[:, y1, x0] if y0 < 0 and x1 == width else (image[:, y1, x1] if y0 < 0 else image[:, y0, x0]))

    data_11 = image[:, y1, x1] if (y1 < height and x1 < width) else (
        image[:, y0, x0] if y1 == height and x1 == width else (image[:, y0, x1] if y1 == height else image[:, y1, x0]))

    data = data_00 * (1 - alpha) * beta + data_01 * (1 - alpha) * (
            1 - beta) + data_10 * alpha * beta + data_11 * alpha * (1 - beta)

    return data


def move_triangles_to_origin(x_space, y_space, triangles):
    h, w, _, _ = triangles.shape
    for y in range(h):
        for x in range(w):
            for i in range(3):
                vertex = triangles[y, x, i]
                a, b = int(vertex[0]), int((vertex[1]))
                vertex[0] += x_space[b, a]
                vertex[1] += y_space[b, a]
                triangles[y, x, i] = vertex
    return triangles


def find_space_by_seams(h_seams, v_seams, image):
    c, h, w = image.shape
    x_space = np.zeros((h, w))
    y_space = np.zeros((w, h))
    h_seams.reverse()
    v_seams.reverse()
    for seam in h_seams:
        for y in range(w):
            for x in range(h):
                if seam[y] - (y_space[y, x] + x) <= 0:
                    y_space[y, x:] += 1
                    break
    for seam in v_seams:
        for y in range(h):
            for x in range(w):
                if seam[y] - (x_space[y, x] + x) <= 0:
                    x_space[y, x:] += 1
                    break
    return x_space, y_space.transpose(1, 0)


def vectorize_image(to_vectorize, triangles, ref_triangles, ref_image):
    h_t, w_t, _, _ = triangles.shape

    for j in range(h_t):
        if j % 100 == 0:
            print(f"vectorizing image,{j / h_t * 100} %")
        for i in range(w_t):
            vertices = triangles[j, i]
            x_range = [min(vertices[0, 0], vertices[1, 0], vertices[2, 0]),
                       max(vertices[0, 0], vertices[1, 0], vertices[2, 0])]
            y_range = [min(vertices[0, 1], vertices[1, 1], vertices[2, 1]),
                       max(vertices[0, 1], vertices[1, 1], vertices[2, 1])]
            for x in range(int(x_range[0]), int(x_range[1] + 1), 1):
                for y in range(int(y_range[0]), int(y_range[1] + 1), 1):
                    center = [x + 0.5, y + 0.5]
                    coordinate = barycentric_coordinates(triangles[j, i], center)
                    if is_inside_triangle(coordinate):
                        ref_coordinate = get_coordinate(coordinate, ref_triangles[j, i])
                        sampled_value = sampleBilinear(ref_image, ref_coordinate)
                        to_vectorize[:, :, y, x] = sampled_value


def plot_grid_triangles(grid, triangles):
    # Plot the grid
    plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(grid[:, :, 0], grid[:, :, 1], 'ko', markersize=1)

    # Plot the triangles
    for row in triangles:
        for triangle in row:
            polygon = Polygon(triangle, closed=True, edgecolor='b', facecolor='none')
            plt.gca().add_patch(polygon)

    plt.xlim(0, grid.shape[0] - 1)
    plt.ylim(0, grid.shape[1] - 1)  # Subtract 1 to adjust for 0-based indexing
    plt.gca().invert_yaxis()  # Invert the y-axis to match typical image coordinates

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Grid and Triangles')
    plt.grid(True)
    plt.savefig('grid_and_triangles.png', dpi=300)
    plt.show()
