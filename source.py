import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as ttf
import torchvision.transforms as transforms
import torch.nn as nn
from urllib.request import urlopen

class ConvDoubled(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(ConvDoubled, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.f(x)

    
class Down(nn.Module):
    def __init__(self, in_channels, down_channels, middle_channels, out_channels):
        super(Down, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_channels, down_channels, 2, stride=2),
            nn.BatchNorm2d(down_channels),
            ConvDoubled(down_channels, middle_channels, out_channels)
        )

    def forward(self, x):
        return self.f(x)


class Up(nn.Module):
    def __init__(self, in_channels, up_channels, pass_channels, middle_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, up_channels, 2, stride=2),
            nn.BatchNorm2d(up_channels)
        ) 
        self.mix = ConvDoubled(up_channels + pass_channels, middle_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.mix(x)
        return x 


class UNetConvDeep7(nn.Module):
    def __init__(self, num_channels=3, num_classes =1):
        super(UNetConvDeep7, self).__init__()
        self.first = ConvDoubled(num_channels, 16, 16)
        self.down1 = Down(16, 32, 32, 32)
        self.down2 = Down(32, 64, 64, 64)
        self.down3 = Down(64, 128, 128, 128)
        self.down4 = Down(128, 256, 256, 256)
        self.down5 = Down(256, 512, 512, 512)
        self.down6 = Down(512, 1024, 1024, 1024)
        self.down7 = Down(1024, 2048, 2048, 2048)
        self.up7 = Up(2048, 1024, 1024, 1024, 1024)
        self.up6 = Up(1024, 512, 512, 512, 512)
        self.up5 = Up(512, 256, 256, 256, 256)
        self.up4 = Up(256, 128, 128, 128, 128)
        self.up3 = Up(128, 64, 64, 64, 64)
        self.up2 = Up(64, 32, 32, 32, 32)
        self.up1 = Up(32, 16, 16, 16, 16)
        self.last = nn.Conv2d(16, num_classes, 3, padding=1)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.down7(x7)
        x = self.up7(x, x7)
        x = self.up6(x, x6)
        x = self.up5(x, x5)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.last(x)
        return x
    
    
def open_image(place):
    if place[:4] == "http":
        place = urlopen(place)
    pic = Image.open(place)
    plt.imshow(pic)
    plt.show()
    return pic


def crop_image(pic, center, scale):
    l = center[0] - scale
    t = center[1] - scale
    pic = ttf.resized_crop(pic, t, l, scale*2, scale*2, 512)
    plt.imshow(pic)
    plt.show()
    return pic


class RotatingImage(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, pic):
        pic = ttf.rotate(pic, self.angle, expand=True)
        return pic
    

class ProcessingImage:
    def __init__(self, 
                 threshold=0.5, 
                 list_transforms=[], device=None, network=None):
        self.threshold = threshold
        self.list_transforms = list_transforms
        self.device = device
        self.network = network
        
    def __call__(self, pic):
        w, h = pic.size
        list_reults = [pic]
            
        num = len(self.list_transforms)
        list_inputs = []
        for (transform,_) in self.list_transforms:
            pic_c = pic.copy()
            pic_c = transform(pic_c)         
            pic_tt = ttf.to_tensor(pic_c)
            pic_tt = pic_tt.unsqueeze(dim=0)
            list_inputs.append(pic_tt)
        inputs_tt = torch.cat(list_inputs).to(self.device)

        self.network.eval()
        with torch.no_grad():
            outputs_tt = self.network(inputs_tt)
        heatmap255_sum_numpy= np.zeros((512, 512))
        for i, (_,transform) in enumerate(self.list_transforms):
            output_tt = outputs_tt[i].squeeze()
            heatmap_tt = torch.sigmoid(output_tt)
            hm_numpy = np.uint8(heatmap_tt.to("cpu").numpy()*255).copy()
            hm_pil = Image.fromarray(hm_numpy, 'L')
            hm_pil = transform(hm_pil)
            hm_numpy = np.asarray(hm_pil)
            heatmap255_sum_numpy += hm_numpy.copy()
            
        heatmap255_ave_numpy = heatmap255_sum_numpy/num
        mask_numpy_low = heatmap255_ave_numpy > self.threshold*255
        mask_numpy = np.zeros((h, w, 3))
        mask_numpy[:,:] = [255, 255, 0]
        mask_numpy[mask_numpy_low] = [255, 0, 255]

        num_pixels = np.count_nonzero(mask_numpy_low)
        mask = Image.fromarray(np.uint8(mask_numpy))
        
        h,w,_ = mask_numpy.shape
        base = Image.new("L", (w, h), 128)
        image_masked = Image.composite(pic, mask, base)
        list_reults.append(image_masked)
        list_reults.append(mask)
            
        return list_reults, num_pixels
        

class SegmentingImage:
    def __init__(self, threshold=0.5, device=None, average=True):
        network = UNetConvDeep7().to(device)
        network.load_state_dict(torch.load("net_dic_0314_05000", map_location=device))
        
        rotate_clockwise = RotatingImage(90)
        rotate_anticlockwise = RotatingImage(-90)
        if average == True:
            list_transforms = [(self.identity, self.identity), 
                               (ttf.vflip, ttf.vflip), 
                               (ttf.hflip, ttf.hflip),
                               (transforms.Compose([ttf.hflip, ttf.vflip]), transforms.Compose([ttf.vflip, ttf.hflip])),
                               (rotate_clockwise, rotate_anticlockwise),
                               (rotate_anticlockwise, rotate_clockwise),
                               (transforms.Compose([rotate_clockwise, ttf.vflip]), transforms.Compose([ttf.vflip, rotate_anticlockwise])),
                               (transforms.Compose([rotate_anticlockwise, ttf.vflip]), transforms.Compose([ttf.vflip, rotate_clockwise]))
                              ]
        else:
            list_transforms = [(self.identity, self.identity)]
        self.segment = ProcessingImage(threshold=threshold, device=device, network=network, list_transforms=list_transforms)
    
    def __call__(self, pic):
        return self.segment(pic)

    def identity(self, pic):
        return pic
