# Central object segmentation by deep learning for fruits and other roundish objects
We present **CROP(Central Roundish Object Painter)**, which identifies and paints the object at the center of an RGB image. Primarily **CROP** works for roundish fruits in various illumination conditions, but surprisingly, it could also deal with images of other organic or inorganic materials, or ones by optical and electron microscopes, although **CROP** was trained solely by 172 images of fruits. The method involves image segmentation by deep learning, and the architecture of the neural network is a deeper version of the original **U-Net**.

To get the idea, see Section 2.2 and 2.3, and to know about processing time series photos see Section 3.3 of  
[https://arxiv.org/pdf/2008.01251.pdf](https://arxiv.org/pdf/2008.01251.pdf).



## Preparation for analysis on photos in local directories and folders. 
Please install Python, PyTorch, Jupyter, Pillow, Matplotlib and then download the following files: `analysis_single.ipynb`, `analysis_multiple.ipynb`, `source3.py` and favorite parameter dictionaries into the same folder, and open `analysis_single.ipynb` or `analysis_multiple.ipynb` by Jupyter Notebook. The instructions on the above parameter dictionaries are found below (git-clone is not enough to get one). 

## How to run **CROP** to analyze individual photos.
The first tab of `analysis_single.ipynb` opend by Jupyter Notebook looks like:
```python
place = r""
name_measurement = r""
dic_name = r""
```
Please specify your directory or folder which contains the photos to be processed as `place`,
the masurement name as `name_measurement` and the parameter dictionary name as `dic_name `.
Note that each measurement will be saved separately in the corresponding directory or folder,
which is useful when you make different experiments on the same photos. 
After executing the first and second tabs, one will get the list of photos with id numbers. 
In the third tab: 
```python
pic_id = 
```
choose a photo by its id number to see it after the execution. 
In the fourth tab:
```python
center = (x ,y)
scale = z
```
please set a cropping frame, the way of which is written below, and execute the fifth tab.
If the cropped photo is fine, execute the next tab.
Note that as default, the program will automatically adjust the cropping frame in the next process. 

As a result, you will see all the processes; as default
automatically adjusted cropping frame, eleven differently scaled photos with masks, 
histogram of the measurement outcomes, and the center of mass in the original photo. 
In addition, the directory or folder named as `name_measurement`, which was defined earlier,
will be created and the program will save the eleven measurement outcomes, the median and the corresponding coordinates of the center of mass in a csv file, and masked images and eleven masks (thumbnails).

If you want to process another photo, make similar operations from the third tab. 
An additional data will be added up in the csv file, and photos may be overwritten if one processes the same photo. 

## How to run **CROP** to analyze time series photos.
The first tab of `analysis_multiple.ipynb` can be handled in the same way as `analysis_single.ipynb`,
and then one can execute the second tab without changing anything unless one wants to change parameters of the program. 
One will see then the first photo in the directory or folder; 
we assume that they are ordered chronologically and alphabetically at the same time.
The third tab can be completed as `analysis_single.ipynb`, again. 
The execution of the fourth tab will give you all; as default
the program will make the directory or folder named as `name_measurement`
and save the same data as `analysis_single.ipynb`, but this time the data for all the photos.
So, the target object can move around in the time series photos, 
but it must be almost at the same position between neighboring photos. 


The instructions below correspond to the first version of the arxiv paper. Nevertheless, one can consult them to make trials on photos on the net, for example. 

## Preparation for tryials on photos on the net, and local ones. 
To use **CROP**, one can install Python, PyTorch, Jupyter, Pillow, Matplotlib. Then, download the following three files: `demo.ipynb`, `source.py` and `net_dic_0314_5000` into the same folder, and open `demo.ipynb` by Jupyter Notebook. 


Before going into the instruction, let us make some remarks. The file `net_dic_0314_5000` is placed as a [release](https://github.com/MotohisaFukuda/CROP/releases), which is the python dictionary object containing parameters for **CROP**. As it is compressed as a zip file, please decompress and use it.  Also one can find other files of such dictionaries. If the name contains `ft`, it was fine-tuned to particular pears in the local farms, otherwise it is meant for general roundish objects. To use them, one needs to replace the dictionary names in `source.py`. 

Also, **CROP** averages eight different outcomes based on dihedral transformations. Therefore, without a GPU some may think the program runs slowly. This extra averaging process can be deactivated as described below. 


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
