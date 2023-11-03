import os

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch

from Gradcam_github.utils import visualize_cam
from Gradcam_github.produce import get_mask
from Gradcam_github.produce import grad_cam


class feature_map_class():
    def __init__(self, original_mask, image):
        self.mask = original_mask
        self.image = image
        _, _, _, self.size = image.shape
        self.selecting = False
        self.circle_center = (0, 0)
        self.circle_radius = self.size / 40
        self.fig, self.ax = plt.subplots()
        self.feature_map = grad_cam(self.mask, self.image)
        self.title_font ={
            'family': 'serif',  # Font family (e.g., 'serif', 'sans-serif', 'monospace')
            'color': 'black',  # Font color
            'weight': 'light',  # Font weight ('normal', 'bold', 'heavy', 'light', etc.)
            'size': 10  # Font size
        }
    def paint_featuremap(self):
        # Create a figure and axis for Matplotlib

        self.ax.imshow(self.feature_map.permute(1, 2, 0))  # Display the tensor as an image

        # Define variables for user input

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_press_event', self.on_button)
        self.fig.canvas.mpl_connect('button_release_event', self.on_button)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.title('Left click and move around on the image to start painting, right click to stop. \n Use arrow keys '
                  '"up" and "down" to resize the brush',fontdict=self.title_font)
        plt.show()
        # Function to create a circular mask

    def create_circle_mask(self, shape, center, radius):
        x, y = np.ogrid[:shape[1], :shape[0]]
        # Calculate the distance of each pixel to the center
        distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        # Normalize the distances to a range from 0 to 1
        mask = 1 - np.clip(distances / radius, 0, 1)

        # Create a mask based on the normalized distances
        return mask

    # Function to handle mouse motion
    def on_motion(self, event):
        y, x = event.ydata, event.xdata
        self.circle_center = (x, y)

        if self.selecting:
            mask = self.create_circle_mask(self.image.shape[-2:],
                                           (self.circle_center[0], self.circle_center[1]),
                                           self.circle_radius)
            mask = torch.from_numpy(mask).T.unsqueeze(0).unsqueeze(0)
            self.mask = torch.clamp((mask + self.mask), 0, 1)
            self.feature_map = grad_cam(self.mask, self.image)
            self.ax.clear()
            plt.title(
                'Left click and move around on the image to start painting, right click to stop. \n Use arrow keys '
                '"up" and "down" to resize the brush\n cancel the plot to continue', fontdict=self.title_font)
            self.ax.imshow(self.feature_map.permute(1, 2, 0))  # Display the color image
            self.ax.add_patch(plt.Circle(self.circle_center, self.circle_radius, color='w', fill=False))
            self.fig.canvas.draw()

    # Function to handle mouse button events
    def on_button(self, event):
        if event.button == 1:  # Left mouse button click
            if event.name == 'button_press_event':
                self.selecting = True
        elif event.button == 3:  # Right mouse button click to clear the selection
            if event.name == 'button_press_event':
                self.selecting = False
                self.ax.clear()
                plt.title(
                    'Left click and move around on the image to start painting, right click to stop. \n Use arrow keys '
                    '"up" and "down" to resize the brush', fontdict=self.title_font)
                self.ax.imshow(self.feature_map.permute(1, 2, 0))  # Display the color image
                self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'up':
            self.circle_radius += 1
            print("circle radius up")
        if event.key == 'down':
            self.circle_radius -= 1
            print("circle radius down.")
