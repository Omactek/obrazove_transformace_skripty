from task_01 import parse_metadata, radiometric_correction
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy.linalg as linalg

image_metadata = r"data\21_03_2022_clipped\LC08_L1TP_192025_20220321_20220329_02_T1_MTL.txt"
data_dir = r"data\21_03_2022_clipped"
dir_path = os.path.dirname(image_metadata)

def create_flat_df(bands):
    bands_flat = []
    for i in range (2,8):
        bands_flat.append(bands[i].flatten())
    bands_array = np.array(bands_flat).T
    cols = [f'Band {i}' for i in range(2, 8)]
    df = pd.DataFrame(bands_array, columns=cols)
    return df

def reshape_flat_pca(bands): #creates 2d array (all pixels * bands)
    bands_fl = [bands[i] for i in range(2,8)]
    stacked_bands = np.stack(bands_fl, axis=-1)
    return stacked_bands.reshape(-1, stacked_bands.shape[-1])

def reshape_pca_results(pca_result, shape):
    return [pca_result[:, i].reshape(shape) for i in range(pca_result.shape[1])]

def scale_vals(img):
    band_min = np.min(img)
    band_max = np.max(img)
    norm = (img - band_min) / (band_max - band_min)
    return np.uint8(255 * norm)

def calc_matrices(df):
    correlation_matrix = df.corr()
    covariance_matrix = df.cov()
    return correlation_matrix, covariance_matrix

def calc_eig(matrix): #isnt used in the code
    eigenvalues, eigenvectors = linalg.eig(matrix)
    return eigenvalues, eigenvectors

def calculate_PCA(df): #expects flattened dataset
    norm_df = (df - df.mean()) / df.std()
    pca = PCA()
    pca_result = pca.fit_transform(norm_df)
    eigenvalues = pca.explained_variance_
    expl_var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(expl_var_ratio)

    pca_info = pd.DataFrame({
        "Component": [f'Comp {i}' for i in range(1, 7)],
        "Eigenvalues": eigenvalues,
        "Variance [%]": expl_var_ratio*100,
        "Cumulative variance [%]": cum_var*100
    })
    
    return eigenvalues, expl_var_ratio, cum_var, pca_info, pca_result

def create_corr_heatmap(matrix):
    my_mask = np.triu(np.ones_like(matrix,dtype=bool))
    plt.figure(figsize=(10,8))
    sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=1, annot=True, square=True, mask=my_mask)
    plt.savefig("corr_matrix", dpi=300, bbox_inches="tight")

def create_cov_heatmap(matrix):
    my_mask = np.triu(np.ones_like(matrix,dtype=bool))
    plt.figure(figsize=(10,8))
    sns.heatmap(matrix, cmap="Blues",annot=True, square=True, mask=my_mask)
    plt.savefig("covar_matrix", dpi=300, bbox_inches="tight")

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

def create_rgb_image(calib_bands):
    bands = []
    for i in range(4,1, -1):
        scaled = scale_vals(calib_bands[i])
        bands.append(scaled)
    rgb_image = np.dstack(bands)
    return rgb_image

if __name__ == '__main__':
    bands, reflect_mult, reflect_add, sun_el = parse_metadata(image_metadata, 1, 7)
    calibrated_bands = radiometric_correction(bands, reflect_mult, reflect_add, sun_el, data_dir, 1, 7)
    flat_data = create_flat_df(calibrated_bands)

    cor_matrix, cov_matrix = calc_matrices(flat_data)

    create_corr_heatmap(cor_matrix)
    create_cov_heatmap(cov_matrix)
    flat_stack = reshape_flat_pca(calibrated_bands)
    eigenvalues, expl_var_ratio, cum_var, pca_info, pca_res = calculate_PCA(flat_stack)
    print(pca_info)
    pca_components = reshape_pca_results(pca_res, calibrated_bands[1].shape)

    x,y = calibrated_bands[1].shape
    pc3_image = create_image_pc(pca_res, x, y)

    plt.figure()
    plt.imshow(pc3_image)
    plt.axis('off')
    plt.title("PCA first 3 components synthesis")
    plt.savefig("pca_first_3_colour_synthesis_8bit.png", dpi=300, bbox_inches='tight')
    plt.close()

    pc3_image = create_image_pc(pca_res, x, y, scale=False)
    plt.figure()
    plt.imshow(pc3_image)
    plt.axis('off')
    plt.title("PCA first 3 components synthesis")
    plt.savefig("pca_first_3_colour_synthesis.png", dpi=300, bbox_inches='tight')
    plt.close()

    original_image = create_rgb_image(calibrated_bands)

    plt.figure()
    plt.imshow(original_image)
    plt.axis('off')
    plt.title("Original RGB Image")
    plt.savefig("orig_img.png", dpi=300, bbox_inches='tight')