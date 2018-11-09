import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
image =sitk.ReadImage("F:\\data\\dataMask.mhd")
image = sitk.GetArrayFromImage(image)
#image = np.squeeze(image[slice, ...])  # if the image is 3d, the slice is integer
plt.imshow(image,cmap='gray')
plt.axis('off')
plt.show()
#cv2.imwrite('1.png',image)
