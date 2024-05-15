import numpy as np
import torch
import math
import cv2
from skimage.metrics import structural_similarity
import torch.nn as nn


# ----------------------------------------convert------------------------------------------



def srgb_to_lin(image):
    thresh = 0.04045
    if torch.is_tensor(image):
        low_val = image <= thresh
        im_out = torch.zeros_like(image)
        im_out[low_val] = 25 / 323 * image[low_val]
        im_out[torch.logical_not(low_val)] = ((200 * image[torch.logical_not(low_val)] + 11)
                                              / 211) ** (12 / 5)
    else:
        im_out = np.where(image <= thresh, image / 12.92, ((image + 0.055) / 1.055) ** (12 / 5))
    return im_out

def lin_to_srgb(image):
    thresh = 0.0031308
    im_out = np.where(image <= thresh, 12.92 * image, 1.055 * (image ** (1 / 2.4)) - 0.055)
    return im_out



def phase_to_img(holo_phase,mean=0):
    # phase to image
    max_phs = 2 * torch.pi
    holo_phase = torch.squeeze(holo_phase)
    if mean==1:
        holo_phase = holo_phase - holo_phase.mean()
    holo_phase = ((holo_phase + max_phs / 2) % max_phs) / max_phs
    #holo_phase = ((holo_phase) % max_phs) / max_phs
    holo_phase = np.uint8(holo_phase.cpu().data.numpy() * 255)
    return holo_phase

def phase_to_img_color(holo_phase):
    # phase to image
    holo0 = phase_to_img(holo_phase[:,0,:,:])
    holo1 = phase_to_img(holo_phase[:,1,:,:])
    holo2 = phase_to_img(holo_phase[:,2,:,:])
    color = cv2.merge([holo0, holo1, holo2])
    return color



# ----------------------------------------read image------------------------------------------


def load_img(path, color_channel=1, resize=0, res=(3840, 2160), to_amp=0,keep_numpy=0):
    # give image path, to torch.cuda
    img = cv2.imread(path)
    if resize == 1:
        img = cv2.resize(img, res)
    if color_channel != 3:
        img = cv2.split(img)[color_channel]
    if keep_numpy == 0:
        img = torch.from_numpy(img)
        if color_channel != 3:
            img = img.unsqueeze(0).unsqueeze(0).cuda()  # x,y to 1,1,x,y
        else:
            img = img.permute(2, 0, 1).contiguous().unsqueeze(0).cuda()  # x,y,3 to 1,3,x,y
        img = img / 255.0
        if to_amp == 1:
            img = srgb_to_lin(img)
            img = torch.sqrt(img)

    return img

