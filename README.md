# Central object segmentation by deep learning for fruits and other roundish objects
We present **CROP(Central Roundish Object Painter)**, which identifies and paints the object at the center of an RGB image. Primarily **CROP** works for roundish fruits in various illumination conditions, but surprisingly, it could also deal with images of other organic or inorganic materials, or ones by optical and electron microscopes, although **CROP** was trained solely by 172 images of fruits. The method involves image segmentation by deep learning, and the architecture of the neural network is a deeper version of the original **U-Net**.


## Preparation
To use **CROP**, one can install **python**, **pytorch**, **torchvision**, **jupyter**, **pillow**, **matplotlib**, perhaps with **conda**. Then, download the following three files: `demo.ipynb`, `source.py` and `net_dic_0314_5000` into the same folder, and open `demo.ipynb` by **Jupyter Notebook**. 


Before going into the instruction, let us make some remark. The file `net_dic_0314_5000` is placed as a [release](https://github.com/MotohisaFukuda/CROP/releases), which is the python dictionary object containing parameters for **CROP**. As it is compressed as a zip file, please decompress and then use it.  Also one can find another file `net_dic_ft_0328_1_5000`, which contains parameters after the fine-tuning to particular pears in the local farms. To use it, one needs to replace the dictionary names in `source.py`. Please see the paper for the fine-tuning:
TBA
As is also written in the paper, **CROP** averages eight different outcomes based on dihedral transformations. Therefore, without a GPU some may think the program runs slowly. This extra averaging process can be deactivated as described below. 


## How to run **CROP** on **Jupyter Notebook**

The first tab looks like:
```python
from source import *
device = "cuda:0" if torch.cuda.is_available() else "cpu"
segment_image = SegmentingImage(threshold=0.5, device=device, average=True)
```
Here, necessary codes will be imported from the file `source.py`. By default, it will use the zero-numbered GPU if available to pytorch. Change the GPU number if you prefer, and execute the tab. In addition, `threshold` can be set to a value between 0 and 1. The smaller it is, the more likely for **CROP** to classify pixels as part of the central object. Besides, one can switch off the extra averaging process by setting `average=False`. 

The second tab is:
```python
place = ""
pic_original = open_image(place)
```
Insert into between double-quotation marks a URL or a path to your image, and then execute the tab. You will see below the tab your image with scales.

By using the third tab:
```python
center = (x, y)
scale = z
pic_cropped = crop_image(pic_original, center, scale)
```
you can crop the image through setting coordinates for the center and choosing the cropping size. Here, `x` and `y` are horizontal and vertical coordinates, respectively, of the center of your cropping frame. Then, the top-left corner will be `(x-z, y-z)` and the bottom-right `(x+z, y+z)`. After executing this tab, you will get a cropped image resized to 512 times 512. You can redo this process until you get your favorite cropped image.

The final tab shown below will give you the segmented image and the number of pixels (out of 512 times 512) which **CROP** thought belong to the central object. Here is the final tab:
```python
list_pics, num_pixels = segment_image(pic_cropped)
for pic in list_pics:
    plt.imshow(pic)
    plt.axis('off')
    plt.show()
print(num_pixels)
```
Note that `list_pics` is the list containing the image from the last tab, the masked image by **CROP**, and the mask itself. Also, `num_pixels` is the number of pixels for the object based on the mask image. They all will be displayed under the tab after execution. 
