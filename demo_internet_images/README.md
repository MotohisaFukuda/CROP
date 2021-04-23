## How to apply **CROP** to images on the net via **Jupyter Notebook**

We explain below how to process images on the net by using the notebook named `demo.ipynb`. You need to place `source.py`, and `net_dic_0314_05000`, which is the default parameter dictionary (we have other parameter dictionaries too). The dictionaries are placed at [release](https://github.com/MotohisaFukuda/CROP/releases), so you can downlaod and unfreeze them. 

The first tab looks like:
```python
from source import *
device = "cuda:0" if torch.cuda.is_available() else "cpu"
segment_image = AnalyzingImage(dic_name="net_dic_0314_05000", device=device, threshold=0.5, average=True, mode="median", save_all=False,
                 show_adjusted=False, show_adjust_process=None, show_final=True, show_in_original=True, 
                 size_pointer1=5, size_pointer2=10)
#SegmentingImage(threshold=0.5, device=device, average=True)
```
Here, necessary codes will be imported from the file `source.py`. By default, it will use the zero-numbered GPU if available to pytorch. Change the GPU number if you prefer, and execute the tab. In addition, `threshold` can be set to a value between 0 and 1. The smaller it is, the more likely it is for **CROP** to classify pixels as part of the central object. Besides, one can switch off the extra averaging process by setting `average=False`. By default, it finds the central roundish object, counts the number of pixels in 11 different scales and picks the median. One can alter these functions by setting `mode` to be `"manual"` or `"center"`.

Execute the second tab:
```python
url = input()
```
and input the url of the image on the net in the popped-up box. 
Then execute the third:
```python
pic_original = open_image(url)
```
So that you will see below the tab your image with scales.

By using the third tab:
```python
center = (x, y)
scale = z
pic_cropped = crop_image(pic_original, center, scale)
```
you can crop the image through setting coordinates for the center and choosing the cropping size. Here, `x` and `y` are horizontal and vertical coordinates, respectively, of the center of your cropping frame. Then, the top-left corner will be `(x-z, y-z)` and the bottom-right `(x+z, y+z)`. After executing this tab, you will get a cropped image. You can redo this process until you set the target object around at the center.

If the cropped photo is fine, execute the next (and last) tab:
```python
_ = segment_image(pic_original, center, scale)
```
the program will automatically adjust the cropping frame,
and show the adjusted cropping frame, eleven differently scaled photos with masks, 
the histogram of the measurement outcomes, and the center of mass in the original photo. 
