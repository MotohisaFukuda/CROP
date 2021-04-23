## How to run **CROP** to analyze time series photos in a directory or folder.

We explain below how to analyze time series photos in a folder. Please read the instruciton for 

The first tab of `analysis_timeseries` is the same as `analysis_single.ipynb`:
```python
place = r""
name_measurement = r""
dic_name = r""
```
Please specify your directory or folder which contains the photos to be processed as `place`,
the masurement name as `name_measurement` and the parameter dictionary name as `dic_name `.
The dictionaries are placed at [release](https://github.com/MotohisaFukuda/CROP/releases), so you can downlaod and unfreeze them.
Note that each measurement outcome will be saved separately in the corresponding directory or folder created under the measurement name.
This function is useful when you make different experiments on the same photos. 
After executing the first and second tabs, you will see then the first photo in the directory or folder; 
we assume that they are ordered chronologically and alphabetically at the same time.
In the third tab:
```python
center = (x ,y)
scale = z
```
you can crop the image through setting coordinates for the center and choosing the cropping size. Here, `x` and `y` are horizontal and vertical coordinates, respectively, of the center of your cropping frame. Then, the top-left corner will be `(x-z, y-z)` and the bottom-right `(x+z, y+z)`. After executing this tab, you will get a cropped image. You can redo this process until you set the target object around at the center.

The execution of the fourth tab will give you all; as default
the program will make the directory or folder named as `name_measurement`
and save the eleven measurement outcomes, the median and the corresponding coordinates of the center of mass in a csv file, and masked images and eleven masks (thumbnails) for all the photos.
The target object can move around little by little in the time series photos, 
but **CROP** can follow it because it can find the roundish objeect almost at the center. 
