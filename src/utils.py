import os
import cv2
import numpy as np 
from pathlib import Path

import skimage.io
import colorsys

DEFAULT_CHANNELS = (1, 2, 3)

RGB_MAP = {
    1: {"rgb": np.array([255, 0, 0]), "range": [0, 60]},
    2: {"rgb": np.array([0, 255, 0]), "range": [0, 255]},
    3: {"rgb": np.array([0, 0, 255]), "range": [0, 100]},
}


def convert_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):
    """
    Converts and returns the image data as RGB image
    Parameters
    ----------
    t : np.ndarray
        original image data
    channels : list of int
        channels to include
    vmax : int
        the max value used for scaling
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.
    Returns
    -------
    np.ndarray the image data of the site as RGB channels
    """
    dim1, dim2, _ = t.shape
    colored_channels = []
    for i, channel in enumerate(channels):
        x = (t[:, :, channel - 1] / vmax) / (
            (rgb_map[channel]["range"][1] - rgb_map[channel]["range"][0]) / 255
        ) + rgb_map[channel]["range"][0] / 255
        x = np.where(x > 1.0, 1.0, x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]["rgb"]).reshape(dim1, dim2, 3), dtype=int
        )
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    im = im.astype(np.uint8)
    return im

def get_composite_image(df, seg_cycle="Cycle1"):
    # Read condition and cycles name
    conditions = df.Location.unique()
    imgs = []

    for condition in conditions:
        df_subset = df[
            (df.Location == condition)
            & (df.Cycle == seg_cycle)
            & (df.Z_stack == "Combined")
        ]

        dapi_path = df_subset[df_subset.Marker == "Hoeschst"].Path_corrected.item()
        pha_path = df_subset[df_subset.Marker == "Phalloidin"].Path_corrected.item()
        wga_path = df_subset[df_subset.Marker == "WGA"].Path_corrected.item()

        img_dapi = skimage.io.imread(dapi_path)
        img_pha = skimage.io.imread(pha_path)
        img_wga = skimage.io.imread(wga_path)

        data = np.concatenate(
            (
                img_pha[:, :, np.newaxis],
                img_wga[:, :, np.newaxis],
                img_dapi[:, :, np.newaxis],
            ),
            axis=2,
        )
        img = convert_to_rgb(data, vmax=65536)
        imgs.append(img)

    return imgs, conditions

def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r,g,b), axis=-1)
    return rgb

def mask_overlay(img, masks, colors=None):
    """ overlay masks on image (set image to grayscale)
    Parameters
    ----------------
    img: int or float, 2D or 3D array
        img is of size [Ly x Lx (x nchan)]
    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels
    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range
    Returns
    ----------------
    RGB: uint8, 3D array
        array of masks overlaid on grayscale image
    """
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)
    if img.ndim>2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(img*1.5, 0, 1.0)
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = np.random.rand()
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = 1.0
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

# Read mask image
def get_masks(mask_folder):
    '''
    Function to get all mask from mask forlder
    '''
    # Read masks
    masks = {}

    for (dirpath, dirnames, filenames) in os.walk(mask_folder):
        for name in sorted(filenames):
            if "tiff" in name:
                condition = name.split("_")[0]
                masks[condition] = masks.get(condition, {})
                filename = os.path.join(dirpath, name)
            if "cyto" in name:
                img = skimage.io.imread(filename)
                masks[condition]["cyto"] = img
            elif "nuclei" in name:
                img = skimage.io.imread(filename)
                masks[condition]["nuclei"] = img
    return masks

# Quality control of mask
def qc_nuclei(mask_cyto, mask_nuclei):
    '''
    Function to check if cell masks contain nuclei
    '''
    cell = np.zeros((mask_cyto.shape), dtype=np.uint8)
    nuclei = np.zeros((mask_cyto.shape), dtype=np.uint8)
    cyto = np.zeros((mask_cyto.shape), dtype=np.uint8)

    for label in range(1, mask_cyto.max()):
        # Check if cell has nuclei
        cell_mask = np.where(mask_cyto == label, 1, 0).astype(np.uint8)
        maski = cv2.bitwise_and(mask_nuclei, mask_nuclei, mask=cell_mask)

        # If no nuclei detected then pass
        if maski.max() == 0:
            continue

        # Link label accross cell, nuclei, cyto
        cell = np.where(mask_cyto == label, label, cell)
        nuclei = np.where(maski > 0, label, nuclei)
        maski = cv2.subtract(cell_mask, maski)
        cyto = np.where(maski > 0, label, cyto)
    return cell, nuclei, cyto
