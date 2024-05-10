import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

x = torch.rand(5, 3)
#print(x)

image_path = r'C:\GitHub\DL_project\dataset\AnnualCrop\AnnualCrop_235.tif'

img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

if img is not None:
    print("Dimensioni dell'immagine:", img.shape)
else:
    print("Impossibile aprire l'immagine.")


#img = Image.open('C:\GitHub\DL_project\dataset\AnnualCrop\example_TIFF_1MB.tif')

#plt.imshow(img)
#plt.show()


