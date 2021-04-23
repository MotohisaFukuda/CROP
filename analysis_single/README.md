## How to run **CROP** to analyze individual photos in a directory or folder.

We explain below how to analyze a single photo among several photos in a folder. 

The first tab of `analysis_single.ipynb` opened by Jupyter Notebook looks like:
```python
place = r""
name_measurement = r""
dic_name = r""
```
Please specify your directory or folder which contains the photos to be processed as `place`,
the masurement name as `name_measurement` and the parameter dictionary name as `dic_name `.
The dictionaries are placed at [release](https://github.com/MotohisaFukuda/CROP/releases), so you can downlaod and unfreeze them.
Note that each measurement outcomes will be saved separately in the corresponding directory or folder created under the measurement name.
This function is useful when you make different experiments on the same photos. 
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
you can crop the image through setting coordinates for the center and choosing the cropping size. Here, `x` and `y` are horizontal and vertical coordinates, respectively, of the center of your cropping frame. Then, the top-left corner will be `(x-z, y-z)` and the bottom-right `(x+z, y+z)`. After executing this tab, you will get a cropped image. You can redo this process until you set the target object around at the center.

If the cropped photo is fine, execute the next (and last) tab.
Note that as default, the program will automatically adjust the cropping frame,
and show the adjusted cropping frame, eleven differently scaled photos with masks, 
the histogram of the measurement outcomes, and the center of mass in the original photo. 
In addition, as was explained above, the directory or folder named as `name_measurement`
will be created and the program will save the eleven measurement outcomes, the median and the corresponding coordinates of the center of mass in a csv file, and masked images and eleven masks (thumbnails).

If you want to process another photo, make similar operations from the third tab. 
An additional data will be added up in the csv file, and photos may be overwritten if one processes the same photo. 
