"""
to find mean/median of array

import numpy as np
a = np.array([14,14,14)]
print(a)
m = np.mean(a)
med = np.median(a)

code to resize

import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('C:////')
print(img.shape)
img2 = cv2.resize(img,(500,500))
plt.imshow(cv2.cvtColor(im2,cv2.COLOR_BGR2RGB))
plt.show()

code to find size min/max value of an image

import cv2
import matplotlib.pyplot as plt
img = cv2.imread('C:////')
print(img.shape)
print(img.min())
print(img.max())
plt.show()

code to display img in greyscale

import cv2
import matplotlib.pyplot as plt
img = cv2.imread('C:////')
plt.imshow(img,cmap = 'gray')
plt.title('birds')
plt.show()

code to display image

import cv2
import matplotlib.pyplot as plt
img = cv2.imread('C:////')
plt.imshow(img)
plt.title('birds')
plt.show()

code to display temp/date

import matplotlib.pyplot as plt
date=['23/4','24/4']
temp=[12,45]
plt.plot/bar(date,temp)
plt.xlabel('Date')
plt.ylabel('Temp')
plt.show()














