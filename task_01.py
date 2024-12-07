import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

metadata = [r"data\05_08_2022_clipped\LC08_L1TP_191025_20220805_20220818_02_T1_MTL.txt", r"data\21_03_2022_clipped\LC08_L1TP_192025_20220321_20220329_02_T1_MTL.txt"]
def parse_metadata(input_file, first_b, last_b):
    bands = {}
    reflect_mult = {}
    reflect_add = {}
    sun_el_key = ""
    band_keys = {f"FILE_NAME_BAND_{i}" for i in range(first_b, last_b+1)}
    reflect_mult_keys = {f"REFLECTANCE_MULT_BAND_{i}" for i in range(first_b, last_b+1)}
    reflect_add_keys = {f"REFLECTANCE_ADD_BAND_{i}" for i in range(first_b, last_b+1)}
    sun_el_key = "SUN_ELEVATION"
    with open(input_file, "r") as f:
        for line in f:
            key = line.split("=")[0].strip()
            if key in band_keys:
                band_num = int(key.split("_")[-1])
                bands[band_num] = line.split("=")[1].strip().strip('"')
            elif key in reflect_mult_keys:
                band_num = int(key.split("_")[-1])
                reflect_mult[band_num] = float(line.split("=")[1].strip())
            elif key in reflect_add_keys:
                band_num = int(key.split("_")[-1])
                reflect_add[band_num] = float(line.split("=")[1].strip())
            elif key == sun_el_key:
                sun_elevation = float(line.split("=")[1].strip())
    return bands, reflect_mult, reflect_add, sun_elevation

def radiometric_correction(band_dict, reflect_mult, reflect_add, sun_el, data_dir, first_b, last_b):
    bands_calib = {}
    for i in range(first_b,last_b+1):
        with rasterio.open(os.path.join(data_dir, band_dict[i])) as src:
            dn_array = src.read(1)
            #TOA reflectance without correction for solar angle
            toa_reflectence = reflect_mult[i] * dn_array + reflect_add[i]

            #TOA reflectance with a correction for the sun angle, currentntly not used
            bands_calib[i] = toa_reflectence#/(np.sin(np.radians((sun_el))))
            #print(f"Band {i}: ")
            #correct_vals = np.sum((bands_calib[i] >= 0) & (bands_calib[i] <= 1))
            #range_perc = (correct_vals/bands_calib[i].size)*100
            #print(f"Min: {np.min(bands_calib[i])}, Max: {np.max(bands_calib[i])}, Mean: {np.mean(bands_calib[i])}, StdDev: {np.std(bands_calib[i])}, in range: {range_perc} %")
    return bands_calib

def cap_transformation(cbd): #input: calibrated band dictionary
    brigthness = (0.0000*cbd[1]	+ 0.3029*cbd[2]	+ 0.2786*cbd[3] + 0.4733*cbd[4] + 0.5599*cbd[5] + 0.5080*cbd[6] + 0.1872*cbd[7])
    greenness = (0.0000*cbd[1] - 0.2941*cbd[2] - 0.2430*cbd[3] - 0.5424*cbd[4] + 0.7276*cbd[5] + 0.0713*cbd[6] - 0.1608*cbd[7])
    wetness	= (0.0000*cbd[1] + 0.1511*cbd[2] + 0.1973*cbd[3] + 0.3283*cbd[4] + 0.3407*cbd[5] - 0.7117*cbd[6] - 0.4559*cbd[7])
    return brigthness, greenness, wetness

def format_date(date):
    year = date[:4]
    month = date[4:6]
    day = date[6:]
    formatted_date = f"{day:02}.{month:02}.{year}"
    return formatted_date

def create_hexbin_plot(x_axis, y_axis, date, gridsize=50, cmap="viridis"):
    formated_date = format_date(date)
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(x_axis, y_axis, gridsize=gridsize, cmap=cmap, mincnt=1)
    plt.colorbar(hb, label="Counts per Bin")
    plt.title(f"Brightness vs Greenness {formated_date}")
    plt.xlabel("Brightness")
    plt.ylabel("Greenness")
    plt.savefig(date, dpi=300, bbox_inches="tight")

def procces_image(metadata_image):
    data_folder = os.path.dirname(metadata_image)
    acquisition_date = os.path.basename(metadata_image).split("_")[4]
    bands, reflect_mult, reflect_add, sun_el = parse_metadata(metadata_image, 1, 7)
    calibrated_bands = radiometric_correction(bands, reflect_mult, reflect_add, sun_el, data_folder,1,7)
    brightness, greenness, wetness = cap_transformation(calibrated_bands)
    create_hexbin_plot(brightness, greenness, acquisition_date ) #hexbin makes more sense for large dataset

if __name__ == '__main__':
    for meta in metadata:
        procces_image(meta)