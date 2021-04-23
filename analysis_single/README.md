## How to run **CROP** to analyze individual photos.

We explain below how to analyze a single photo among several photos in a folder. 

The first tab of `analysis_single.ipynb` opend by Jupyter Notebook looks like:
```python
place = r""
name_measurement = r""
dic_name = r""
```
Please specify your directory or folder which contains the photos to be processed as `place`,
the masurement name as `name_measurement` and the parameter dictionary name as `dic_name `.
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
please set a cropping frame, the way of which is written below, and execute the fifth tab.
If the cropped photo is fine, execute the next tab.
Note that as default, the program will automatically adjust the cropping frame in the next process. 

As a result, you will see all the processes; as default
automatically adjusted cropping frame, eleven differently scaled photos with masks, 
histogram of the measurement outcomes, and the center of mass in the original photo. 
In addition, as was explained above, the directory or folder named as `name_measurement`
will be created and the program will save the eleven measurement outcomes, the median and the corresponding coordinates of the center of mass in a csv file, and masked images and eleven masks (thumbnails).

If you want to process another photo, make similar operations from the third tab. 
An additional data will be added up in the csv file, and photos may be overwritten if one processes the same photo. 
