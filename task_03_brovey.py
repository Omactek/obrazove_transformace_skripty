from task_01 import parse_metadata, radiometric_correction
from task_02 import create_rgb_image
import numpy as np
import os
import matplotlib.pyplot as plt

image_metadata = r"data\21_03_2022_clipped\LC08_L1TP_192025_20220321_20220329_02_T1_MTL.txt"
data_dir = r"data\21_03_2022_clipped"
dir_path = os.path.dirname(image_metadata)

def scale_vals(img):
    band_min = np.min(img)
    band_max = np.max(img)
    norm = (img - band_min) / (band_max - band_min)
    return np.uint8(255 * norm)

def normalize_band(band):
    return (band - np.min(band)) / (np.max(band) - np.min(band))

def upscale_image(image, scale_factor):
    return np.repeat(np.repeat(image, scale_factor, axis=0), scale_factor, axis=1)

def upscale_to_pan(bands):
    red = upscale_image(bands[4], 2)
    green = upscale_image(bands[3], 2)
    blue = upscale_image(bands[2], 2)
    scaled_orig = [scale_vals(red), scale_vals(green), scale_vals(blue)]
    scaled_original = np.dstack(scaled_orig)
    return red, green, blue, scaled_original

if __name__ == '__main__':
    bands, reflect_mult, reflect_add, sun_el = parse_metadata(image_metadata, 1, 9)
    calibrated_bands = radiometric_correction(bands, reflect_mult, reflect_add, sun_el, data_dir, 1, 9)

    red, green, blue, scaled_original = upscale_to_pan(calibrated_bands)
    pan = calibrated_bands[8]

    ratio = red + green + blue

    brovey_r = (red / ratio) * pan
    brovey_g = (green / ratio) * pan
    brovey_b = (blue / ratio) * pan

    brp = [scale_vals(brovey_r), scale_vals(brovey_g), scale_vals(brovey_b)]
    broovey = np.dstack(brp)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    original_img = create_rgb_image(calibrated_bands)
    plt.imshow(original_img)
    plt.title("Original RGB Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(broovey)
    plt.title("Brovey Pansharpened Image")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("brovey_pansharpened.png", dpi=300, bbox_inches='tight')