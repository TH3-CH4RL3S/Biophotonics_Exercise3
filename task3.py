import os
import numpy as np
from PIL import Image
import spectral
import wx
import matplotlib.pyplot as plt

def load_spectral_images(output_dir):
    image_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    images = []
    for file in image_files:
        img = Image.open(os.path.join(output_dir, file))
        images.append(np.array(img))
    datacube = np.stack(images, axis=-1)
    return datacube

def show_datacube(datacube):
    app = wx.App(False)  # Create a new wx.App object
    spectral.view_cube(datacube)
    app.MainLoop()  # Start the wx event loop

output_dir = 'output_images3'
datacube = load_spectral_images(output_dir)
show_datacube(datacube)