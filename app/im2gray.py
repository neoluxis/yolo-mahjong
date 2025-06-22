import cv2 as cv
import os
from tqdm.rich import tqdm

path = "/home/neolux/workspace/SmartMahjong/images_grey/"

for file in tqdm(os.listdir(path)):
    if file.endswith(".jpg") or file.endswith(".png"):
        img_path = os.path.join(path, file)
        img = cv.imread(img_path)
        if img is not None:
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray_img = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)  # Convert back to BGR for saving
            cv.imwrite(img_path, gray_img)
            print(f"Converted {file} to grayscale.")
        else:
            print(f"Failed to read {file}.")
    else:
        print(f"Skipping {file}, not an image file.")

