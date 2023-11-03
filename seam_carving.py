import numpy as np
import torch


def find_low_energy_seam(energy_map):
    height, width = energy_map.shape
    dp = np.zeros_like(energy_map)

    dp[0] = energy_map[0]

    for row in range(1, height):
        for col in range(width):
            if col == 0:
                dp[row, col] = energy_map[row, col] + min(dp[row - 1, col], dp[row - 1, col + 1])
            elif col == width - 1:
                dp[row, col] = energy_map[row, col] + min(dp[row - 1, col - 1], dp[row - 1, col])
            else:
                dp[row, col] = energy_map[row, col] + min(dp[row - 1, col - 1], dp[row - 1, col], dp[row - 1, col + 1])
    # to keep the triangle on the frame, the edges will not be carved
    dp[:, width - 1] += 100
    dp[:, 0] += 100
    # dp[height - 1, :] = 100
    # dp[0, : ] += 100
    # Find the seam with the minimum cumulative energy in the last row
    min_energy_col = np.argmin(dp[-1])
    seam = [min_energy_col]
    for row in range(height - 1, 0, -1):
        col = seam[-1]
        if col == 0:
            neighbor = np.argmin([dp[row - 1, col], dp[row - 1, col + 1]])
            seam.append(col + neighbor)
        elif col == width - 1:
            neighbor = np.argmin([dp[row - 1, col - 1], dp[row - 1, col]])
            seam.append(col + neighbor - 1)
        else:
            neighbor = np.argmin([dp[row - 1, col - 1], dp[row - 1, col], dp[row - 1, col + 1]])
            seam.append(col + neighbor - 1)

    # Reverse the seam list to get the correct order from top to bottom
    seam.reverse()

    return seam,np.min(dp[-1])


def remove_seam(seam, image, mask):
    b, c, h, w = image.shape
    new_w = w - 1

    new_image = np.zeros((b, c, h, new_w), dtype=image.dtype)
    new_mask = np.zeros((1, 1, h, new_w), dtype=mask.dtype)

    for row in range(h):
            col = seam[row]  # Column to remove for this row
            new_image[:, :, row, :] = np.delete(image[:, :, row, :], col, axis=-1)
            new_mask[0, 0, row, :] = np.delete(mask[0, 0, row, :], col, axis=-1)

    return new_image, new_mask

def mark_red(seam, image):
    marked_image = image.copy()

    height = marked_image.shape[2]

    red_color = np.array([255, 0, 0], dtype=np.uint8)
    for row in range(height):
        col = seam[row]
        marked_image[0, :, row, col] = red_color  # Mark the pixel in red

    return marked_image
