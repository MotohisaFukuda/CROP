import glob, os, re, csv, copy, time
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as ttf
import torchvision.transforms as transforms
import torch.nn as nn
from urllib.request import urlopen

from IPython.display import clear_output


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
    pic = Image.open(place).convert('RGB')
    plt.imshow(pic)
    plt.show()
    return pic


def crop_image(pic, center, scale, show=True):
    l = center[0] - scale
    t = center[1] - scale
    pic = ttf.resized_crop(pic, t, l, scale*2, scale*2, 512)
    if show:
        plt.imshow(pic)
        plt.show()
    return pic

def crop_image_raw(pic, center, scale, show=True):
    l = center[0] - scale
    t = center[1] - scale
    pic = ttf.crop(pic, t, l, scale*2, scale*2)
    if show:
        plt.imshow(pic)
        plt.axis('off')
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
        
    def __call__(self, pic, adjusting=False, cs=None):
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
        mask_numpy_mono = heatmap255_ave_numpy > self.threshold*255
        y_info, x_info = np.nonzero(mask_numpy_mono)
        
        if adjusting:
            x_min, x_max = np.amin(x_info), np.amax(x_info)
            y_min, y_max = np.amin(y_info), np.amax(y_info)
            
            center, scale = cs
            ratio = scale/256
            center_adjusted = np.around([(x_max + x_min)*ratio/2, (y_max + y_min)*ratio/2]) + np.array(center) - scale
            scale_adjusted = np.around(np.amax([x_max - x_min, y_max - y_min]) * 0.55*ratio)
            return center_adjusted, scale_adjusted
        
        else:
        
            mask_numpy = np.zeros((h, w, 3))
            mask_numpy[:,:] = [255, 255, 0]
            mask_numpy[mask_numpy_mono] = [255, 0, 255]

            num_pixels = np.count_nonzero(mask_numpy_mono)
            mask = Image.fromarray(np.uint8(mask_numpy))

            h,w,_ = mask_numpy.shape
            base = Image.new("L", (w, h), 128)
            image_masked = Image.composite(pic, mask, base)
            list_reults.append(image_masked)
            list_reults.append(mask)

            center_mass = np.mean(x_info), np.mean(y_info)

            return list_reults, num_pixels, center_mass
        

class SegmentingImage:
    def __init__(self, threshold=0.5, device=None, average=True, dic_name=""):
        network = UNetConvDeep7().to(device)
        network.load_state_dict(torch.load(dic_name, map_location=device))
        
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
    
    def __call__(self, pic, adjusting=False, cs=None):
        return self.segment(pic, adjusting, cs)

    def identity(self, pic):
        return pic
    
class AnalyzingImage:
    def __init__(self, dic_name="", device=None, threshold=0.5, average=True, mode="median", save_all=False,
                 show_adjusted=False, show_adjust_process=False, show_final=False, show_in_original=False, 
                 size_pointer1=5, size_pointer2=10):
 
        self.segment_image = SegmentingImage(threshold=threshold, device = device, average = average, dic_name=dic_name)
    
        self.mode = mode
        self.save_all = save_all    
        
        self.show_adjusted = show_adjusted
        self.show_adjust_process = show_adjust_process
        self.show_final = show_final
        self.show_in_original = show_in_original
        
        self.size_pointer1 = size_pointer1
        self.size_pointer2 = size_pointer2
        
        if self.mode == "manual":
            self.process = self.adjust_manually
            
        elif self.mode == "median":
            self.process = self.adjust_automatically
            self.rates=np.round(np.arange(1.0, 2.1, 0.1), decimals=1)
            
        elif self.mode == "center":
            self.process = self.adjust_position_only
        
    def __call__(self, pic_original, center, scale):
        return self.process(pic_original, center, scale)

    def draw_point(self, pic, cm_back_np, size_pointer=5, show=False):
        pic_annotated = pic.copy()
        draw = ImageDraw.Draw(pic_annotated)
        x0, y0 = cm_back_np - size_pointer
        x1, y1 = cm_back_np + size_pointer
        draw.rectangle([x0, y0, x1, y1], fill="blue")
        del draw
        if show:
            plt.imshow(pic_annotated)
            plt.axis('off')
            plt.show()
        return pic_annotated

    def adjust_manually(self, pic_original, center_adjusted, scale_adjusted):
        pic_cropped = crop_image(pic_original, center_adjusted, scale_adjusted, show=False)
        list_pics, num_pixels, center_mass = self.segment_image(pic_cropped)
        cm_np = np.array(center_mass)
        list_pics[1] = self.draw_point(list_pics[1], np.round(cm_np), size_pointer=self.size_pointer1)     
        
        num_pixels_back = num_pixels*((scale_adjusted / 256)**2)
        cm_back = cm_np * (scale_adjusted / 256) + center_adjusted - scale_adjusted

        
        if self.show_final:
            print("pixels:", num_pixels_back)
            plt.imshow(list_pics[1])
            plt.axis('off')
            plt.show()
        if self.show_in_original:
            self.draw_point(pic_original, np.round(cm_back), size_pointer=self.size_pointer2, show=True)
        
        return list_pics, num_pixels_back, cm_back, None, None, None
    
    def adjust_automatically(self, pic_original, center, scale):
        pic_cropped_pre_resized = crop_image(pic_original, center, scale, show=False)
        center_adjusted, scale_adjusted = self.segment_image(pic_cropped_pre_resized, adjusting=True, cs=(center, scale))
        pic_cropped_double = crop_image_raw(pic_original, center_adjusted, 2* scale_adjusted, show=False)
        
        if self.show_adjusted:
            plt.imshow(crop_image_raw(pic_original, center_adjusted, scale_adjusted, show=False))
            plt.axis('off')
            plt.show()
        
        measurements = []
        pics_processed = []
        points_com = []

        for r in self.rates:
            pic_cropped = crop_image(pic_cropped_double, (2*scale_adjusted, 2*scale_adjusted), scale_adjusted*r, show=False)
            list_pics, num_pixels, center_mass = self.segment_image(pic_cropped)

            num_pixels_back = num_pixels*((scale_adjusted*r / 256)**2)
            measurements.append(num_pixels_back)
            pics_processed.append(list_pics)            
            
            if self.show_adjust_process != False:
                print("rate:", r, "pixcels:", num_pixels_back)
                plt.imshow(list_pics[1])
                plt.axis('off')
                plt.show()
                if self.show_adjust_process == "flash":
                    clear_output(wait=True)
                   

            points_com.append(center_mass)
        
        result_sorted = sorted([(i, p) for i, p in enumerate(copy.deepcopy(measurements))], key= lambda x: x[1])
        result_sorted_rp = [(self.rates[i], p) for i, p in result_sorted]
        
        ind_central, pixcels_central = result_sorted[4]
        cm_central_np = np.array(points_com[ind_central])
        rate_central = self.rates[ind_central]
        pic_selected_triple = pics_processed[ind_central]
        pic_selected_triple[1] = self.draw_point(pic_selected_triple[1], np.round(cm_central_np), size_pointer=self.size_pointer1)
        cm_back = cm_central_np * (scale_adjusted*rate_central / 256) + center_adjusted - scale_adjusted*rate_central
        
        
        if self.show_final:
            print("central estimate")
            print("rate", rate_central)
            print("number of pixels", pixcels_central)
            plt.imshow(pic_selected_triple[1])
            plt.axis('off')
            plt.show()
            
            plt.hist(measurements)
#             plt.title('histogram') 
            plt.xlabel('the number of pixels') 
            plt.ylabel('count') 
            plt.show() 

        if self.show_in_original:
            self.draw_point(pic_original, np.round(cm_back), size_pointer=self.size_pointer2, show=True)
            
        if self.save_all:
            pics_masked_all = [pics[1] for pics in pics_processed]
        else:
            pics_masked_all = None
        
        return (pic_selected_triple, pixcels_central, cm_back, 
                rate_central, (center_adjusted, scale_adjusted), (measurements, pics_masked_all)
               )
    
        
    def adjust_position_only(self, pic_original, center, scale):
        pic_cropped_pre_resized = crop_image(pic_original, center, scale, show=False)
        center_adjusted, _ = self.segment_image(pic_cropped_pre_resized, adjusting=True, cs=(center, scale))
        pic_cropped = crop_image(pic_original, center_adjusted, scale, show=False)
        
        if self.show_adjusted:
            plt.imshow(pic_cropped)
            plt.axis('off')
            plt.show()
            
        list_pics, num_pixels, center_mass = self.segment_image(pic_cropped)
        cm_np = np.array(center_mass)
        list_pics[1] = self.draw_point(list_pics[1], np.round(cm_np), size_pointer=self.size_pointer1)     
        
        num_pixels_back = num_pixels*((scale / 256)**2)
        cm_back = cm_np * (scale / 256) + center_adjusted - scale

        
        if self.show_final:
            print("pixels:", num_pixels_back)
            plt.imshow(list_pics[1])
            plt.axis('off')
            plt.show()
        if self.show_in_original:
            self.draw_point(pic_original, np.round(cm_back), size_pointer=self.size_pointer2, show=True)
        
        return list_pics, num_pixels_back, cm_back, None, (center_adjusted, scale), None
        
def find_pics(directory):
    extensions = r"/*.(jpg|jpeg|png|bmp)"
    path = directory
    l = sorted([f for f in os.listdir(path) if re.search(extensions, f, re.IGNORECASE)])
    for i, pic_name in enumerate(l):
        print("id:", i, " name:", pic_name)
    return [(os.path.splitext(name)[0], path + r"/" +name) for name in l]
        
        
        
class ProcessingSinglePic(AnalyzingImage):
    def __init__(self, dic_name="", device=None, directory=None, name_measurement=None, 
                 threshold=0.5, average=True, mode="median", save_all=False, 
                 show_adjusted=False, show_adjust_process=False, show_final=False, show_in_original=None, 
                 size_pointer1=5, size_pointer2=10, mask=False):
        super().__init__(dic_name, device, 
                         threshold, average, mode, save_all, 
                         show_adjusted, show_adjust_process, show_final, show_in_original, 
                         size_pointer1, size_pointer2)

        self.directory_processed = os.path.join(directory, "measurement_" + name_measurement)
        if not os.path.exists(self.directory_processed):
            os.makedirs(self.directory_processed)
        self.path_data_processed = os.path.join(self.directory_processed, "data.csv")
        if self.save_all:
            self.directory_processed_extra = os.path.join(directory, "measurement_" + name_measurement+"_extra")
            if not os.path.exists(self.directory_processed_extra):
                os.makedirs(self.directory_processed_extra)
            self.path_data_processed_extra = os.path.join(self.directory_processed_extra, "data.csv")
        if mask == False:
            self.w = 512 *2
            self.n = 2
        else:
            self.w = 512 *3
            self.n = 3 
        
    def __call__(self, name_path_tuple, center, scale):

        name = name_path_tuple[0]
        pic = Image.open(os.path.join(name_path_tuple[1]))

        list_pics, num_pixels, cm, rate, cs, data_all = self.process(pic, center, scale)
        canvas = Image.new('RGB', (self.w, 512))
        for i, pic in enumerate(list_pics[:self.n]):
            canvas.paste(pic,(512 *i, 0))
        path_pic_processed = os.path.join(self.directory_processed, name+".png")
        canvas.save(path_pic_processed, "png")

        with open (self.path_data_processed, "a") as fil:
            writer = csv.writer(fil)
            writer.writerow([name, num_pixels, cm[0], cm[1], rate])
            
        if self.save_all:
            for r, p in zip(self.rates, data_all[1]):
                path_pic_processed_thumb = os.path.join(self.directory_processed_extra, name+"_"+str(int(10*r))+".png")
                p.thumbnail((128,128))
                p.save(path_pic_processed_thumb, "png")
            with open (self.path_data_processed_extra, "a") as fil:
                writer = csv.writer(fil)
                writer.writerow([name]+data_all[0])
                
        return cs
        
class ProcessingMultiplePics(ProcessingSinglePic):
    def __init__(self, dic_name="", device=None, directory=None, name_measurement=None, 
                 threshold=0.5, average=True, mode="median", save_all=False, 
                 show_adjusted=False, show_adjust_process=False, show_final=False, show_in_original=None, 
                 size_pointer1=5, size_pointer2=10, 
                 exploration_rate=1.5):
        self.exploration_rate = exploration_rate
        super().__init__(dic_name, device, directory, name_measurement,
                         threshold, average, mode, save_all, 
                         show_adjusted, show_adjust_process, show_final, show_in_original, 
                         size_pointer1, size_pointer2)
        
    def __call__(self, name_path_tuples, center, scale):
        scales = [scale]*5
        for name_path_tuple in name_path_tuples:
            cs = super().__call__(name_path_tuple, center, scale)
            center, scale = cs[0], int(cs[1]*self.exploration_rate)
        print("Done!")
        


   
   
    
    
    
    
    
    
    
