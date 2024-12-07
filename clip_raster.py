import rasterio
from rasterio.windows import Window
import os

image_dir = r"data\05_08_2022"
clip_file = r"data\clip_area.shp"
output_dir = r"data\05_08_2022_clipped"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

clip_size = 4000

def open_clip_write(in_data, out_data, clip_size_meters):
    with rasterio.open(in_data) as src:
        width, height = src.width, src.height
        transform = src.transform
        pixel_size_x, pixel_size_y = transform[0], -transform[4]
        
        center_x = width // 2
        center_y = height // 2
        

        clip_size_pixels_x = int(clip_size_meters / abs(pixel_size_x))
        clip_size_pixels_y = int(clip_size_meters / abs(pixel_size_y))
        
        start_x = max(center_x - clip_size_pixels_x // 2, 0)
        start_y = max(center_y - clip_size_pixels_y // 2, 0)
        window = Window(start_x, start_y, clip_size_pixels_x, clip_size_pixels_y)
        
        clipped_data = src.read(window=window)

        clipped_transform = src.window_transform(window)
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height": clip_size_pixels_y,
            "width": clip_size_pixels_x,
            "transform": clipped_transform
        })
        
        with rasterio.open(out_data, "w", **clipped_meta) as dst:
            dst.write(clipped_data)

for file_name in os.listdir(image_dir):
    if file_name.endswith(".TIF"):
        in_path = os.path.join(image_dir, file_name)
        out_path = os.path.join(output_dir, file_name)

        open_clip_write(in_path, out_path, clip_size)
