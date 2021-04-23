# Central object segmentation by deep learning for fruits and other roundish objects
We present **CROP(Central Roundish Object Painter)**, which identifies and paints the object at the center of an RGB image. Primarily **CROP** works for roundish fruits in various illumination conditions, but surprisingly, it could also deal with images of other organic or inorganic materials, or ones by optical and electron microscopes, although **CROP** was trained solely by 172 images of fruits. The method involves image segmentation by deep learning, and the architecture of the neural network is a deeper version of the original **U-Net**.

To get the idea, see Section 2.2 and 2.3, and to know about processing time series photos see Section 3.3 of
arXiv:2008.01251 [cs.CV]:  
[PDF](https://arxiv.org/pdf/2008.01251.pdf), or [Abstract](http://arxiv.org/abs/2008.01251).  

<img src="/images/murayama35a.png" width="33%" /><img src="/images/murayama21a.png" width="33%" /><img src="/images/murayama45.png" width="33%"/> 
<i>Various fruits (photo credit: Hideki Murayama).</i>


## Preparation for analysis on photos in local directories and folders. 
Please install Python, PyTorch, Jupyter, Pillow, Matplotlib and then download the following files: `analysis_single.ipynb`, `analysis_multiple.ipynb`, `source3.py` and favorite parameter dictionaries into the same folder, and open `analysis_single.ipynb` or `analysis_multiple.ipynb` by Jupyter Notebook. The instructions on the above parameter dictionaries are found below (git-clone is not enough to get one). 

## Three different ways of using **CROP**

- [analyze_internet_images.](/demo_internet_images/README.md)

- [analyze individual photos in a directory or folder.](/analysis_single/README.md)

- [analyze time series photos in a directory or folder.](/analysis_timeseries/README.md)

The network dictionaries are placed at [release](https://github.com/MotohisaFukuda/CROP/releases), so you can downlaod and unfreeze them.
To see the difference between dictionaries, please have a look at the paper. 
