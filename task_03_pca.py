from task_01 import parse_metadata, radiometric_correction
from task_03_brovey import upscale_to_pan
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

image_metadata = r"data\21_03_2022_clipped\LC08_L1TP_192025_20220321_20220329_02_T1_MTL.txt"
data_dir = r"data\21_03_2022_clipped"
dir_path = os.path.dirname(image_metadata)

"""def calc_cov_matrix(bands, band_num): 
    flat_bands = []
    for num in band_num:
        flat_bands.append(bands[num].flatten())
    stacked = np.stack(flat_bands)
    cov = np.cov(stacked) #expects 2D array; pixels * bands
    return cov """

def flatten_data(bands, band_num): 
    flat_bands = []
    for num in band_num:
        flat_bands.append(bands[num].flatten())
        print(bands[num].flatten().shape)
    stacked = np.stack(flat_bands)
    return stacked

def normalize(data):
    norm_data = (data - data.mean()) / data.std()
    return norm_data

def upscale_image(image, scale_factor):
    return np.repeat(np.repeat(image, scale_factor, axis=0), scale_factor, axis=1)

def upscale_flatten(bands, band_num):
    flat_bands = []
    for num in band_num:
        upscaled_img = upscale_image(bands[num], 2)
        flat_bands.append(upscaled_img.flatten())
    stacked = np.stack(flat_bands)
    return stacked

def upscale(bands, bands_id, scaler):
    scaled = []
    for num in bands_id:
        upscaled_img = upscale_image(bands[num], scaler)
        scaled.append(upscaled_img)
    return scaled

def flatten(bands, bands_id):
    flat = []
    for num in range(len(bands_id)):
        flattened = bands[num].flatten()
        flat.append(flattened)
    return flat

def calculate_PCA(data): #expects flattened dataset; (pixels x bands)
    norm_data = normalize(data)
    pca = PCA()
    pca_result = pca.fit_transform(norm_data)
    return pca, pca_result

def scale_vals(img):
    band_min = np.min(img)
    band_max = np.max(img)
    norm = (img - band_min) / (band_max - band_min)
    return np.uint8(255 * norm)

def create_image_pc(pca_res, n_rows, n_cols, scale=True):
    pc_images = []
    for i in range(3):
        pc = pca_res[:, i]
        pc_img = pc.reshape(n_rows, n_cols)
        if scale==True:
            pc_rgb = scale_vals(pc_img)
            pc_images.append(pc_rgb)
        else:
            pc_images.append(pc_img)
    pc3_image = np.dstack(pc_images)
    return pc3_image

def reshape_pca_results(pca_result, shape):
    return [pca_result[:, i].reshape(shape) for i in range(pca_result.shape[1])]


bands, reflect_mult, reflect_add, sun_el = parse_metadata(image_metadata, 1, 9)
calibrated_bands = radiometric_correction(bands, reflect_mult, reflect_add, sun_el, data_dir, 1, 9)
#red, green, blue, scaled_original = upscale_to_pan(calibrated_bands)
pan = calibrated_bands[8]
rgb = [2,3,4]

flat_rgb = upscale_flatten(calibrated_bands, rgb)
upscaled_rgb = np.stack(upscale(calibrated_bands, rgb, 2))
print(f"upscaled_rgb: {upscaled_rgb.shape}")
flat_upscaled = np.stack(flatten(upscaled_rgb, rgb))
print(f"flat_upscaled: {flat_upscaled.shape}")

flat_upscaled_t = flat_upscaled.T #PCA needs pixels x bands
scaler = StandardScaler()

# Fit on training set only.
#mnist.data = scaler.fit_transform(mnist.data)
print(f"flat_upscaled_t: {flat_upscaled_t.shape}")

pca = PCA() #create PCA instance
lower_dimensional_data = pca.fit_transform(flat_upscaled_t)
print(f"lower_dimensional_data: {lower_dimensional_data.shape}")
print(f"pca components: {pca.n_components_}")
approximation = pca.inverse_transform(lower_dimensional_data)
print(f"approximation: {approximation.shape}")

plt.figure(figsize=(8,4))

# Original Image
plt.figure(figsize=(12, 6))

def create_rgb_image(calib_bands):
    bands = []
    for i in range(2,-1, -1):
        scaled = scale_vals(calib_bands[i])
        bands.append(scaled)
    rgb_image = np.dstack(bands)
    return rgb_image

def aprox_reshape(data): #entering (70756, 3)
    shaped_list = []
    for i in range(3):
        reshaped = data[i].reshape(266,266)
        print(reshaped.shape)
        shaped_list.append(scale_vals(reshaped))
    return np.dstack(shaped_list)


plt.subplot(1, 2, 1)
plt.imshow(create_rgb_image(upscaled_rgb))
plt.title("Original RGB Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(aprox_reshape(approximation.T))
plt.title("Reconstructed Image")
plt.axis('off')
plt.tight_layout()
plt.savefig("aprox.png", dpi=300, bbox_inches='tight')