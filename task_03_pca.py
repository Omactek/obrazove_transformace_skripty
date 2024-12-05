from task_01 import parse_metadata, radiometric_correction
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
#https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Image_Reconstruction_and_such.ipynb - tutorial code

image_metadata = r"data\21_03_2022_clipped\LC08_L1TP_192025_20220321_20220329_02_T1_MTL.txt"
data_dir = r"data\21_03_2022_clipped"
rgb_id = [2, 3, 4] 
pan_id = [8]

def upscale_bands(bands, bands_id, target_shape):
    bands_ups = []
    for id in bands_id:
        band_ups = resize(bands[id], target_shape, mode='reflect', anti_aliasing=True) #default: bilinear interpolation
        bands_ups.append(band_ups)
    return bands_ups

def normalize_band(band):
    norm = (band - np.min(band)) / (np.max(band) - np.min(band))
    return norm

def scale_vals(band):
    norm = normalize_band(band)
    return np.uint8(255 * norm)

def create_rgb_image(bands, orig_img=False):
    bands_scl = []
    if orig_img==False:
        for i in range(2, -1, -1): #reverse order to create RGB img
            scaled = scale_vals(bands[i])
            bands_scl.append(scaled)
    else:
        for i in range(4, 1, -1): #reverse order to create RGB img
            scaled = scale_vals(bands[i])
            bands_scl.append(scaled)        
    return np.dstack(bands_scl)

bands, reflect_mult, reflect_add, sun_el = parse_metadata(image_metadata, 1, 9)
calibrated_bands = radiometric_correction(bands, reflect_mult, reflect_add, sun_el, data_dir, 1, 9)
pan = calibrated_bands[8]
original_image = create_rgb_image(calibrated_bands, orig_img=True)
pan_shape = pan.shape


upscaled_rgb = np.stack(upscale_bands(calibrated_bands, rgb_id, pan_shape)) #upscale and stack arrays to create image
flat_upsc_rgb_t = upscaled_rgb.reshape(3, -1).T  #expected scikit pca shape (pixels x bands)

scaler = StandardScaler()
normalized_rgb_prepped = scaler.fit_transform(flat_upsc_rgb_t) #standardize

pca = PCA()
lower_dimensional_data = pca.fit_transform(normalized_rgb_prepped) #performs pca transformation

pan_norm = normalize_band(pan).flatten()

pc1 = lower_dimensional_data[:, 0]
pc1_mean, pc1_std, pan_mean, pan_std = pc1.mean(), pc1.std(), pan_norm.mean(), pan_norm.std()
adjusted_pan = (pan_norm - pan_mean) * (pc1_std / pan_std) + pc1_mean #díky Jáchyme <3, adjusts pan band to match mean and std of pc1

lower_dimensional_data[:, 0] = adjusted_pan #replaces pc1 with adjusted pan

approximation = pca.inverse_transform(lower_dimensional_data) #performs reverse pca transformation
reconstructed = scaler.inverse_transform(approximation).T  #reverses scaling
reconstructed_reshaped = reconstructed.reshape(3, *pan_shape)

band_means = np.mean(upscaled_rgb, axis=(1, 2)) #restores original intensity range

reconstructed_with_means = reconstructed_reshaped + band_means[:, None, None] #restores original intensity
pansharpened_image = create_rgb_image(reconstructed_with_means)

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original RGB Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pansharpened_image, interpolation='bicubic')
plt.title("PCA Pansharpened")
plt.axis('off')

plt.tight_layout()
plt.savefig("pca_pansharpened.png", dpi=300, bbox_inches='tight')