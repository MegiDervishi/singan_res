import SinGAN.functions as functions
from SinGAN.imresize import imresize
import torch
import cv2

def read_mask(opt):
    img_name, suffix = opt.input_name.split('.')
    filename = f"{opt.mask_dir}/{img_name}_mask.{suffix}"
    mask = cv2.imread(filename)
    mask = torch.from_numpy(1-mask/255)
    mask = mask.permute((2, 0, 1))[None, :, :, :]
    mask = imresize(mask, opt.scale1, opt)
    return functions.creat_reals_pyramid(mask, [], opt)
    #mask = mask[:, :, :, None]   (0, 1, 2, 3) -> (3, 2, 0, 1)
    #mask = mask.permute((3, 2, 0, 1))

def feed_net(net, x, s, opt, detach = False):
    if opt.inpainting:
        assert x.shape == opt.masks[s].shape, 'Mask and input must be the same shape.'
        if detach:
            return net(x.detach() * opt.masks[s].detach())
        else:
            return net(x * opt.masks[s]).to(opt.device)
    else:
        if detach:
            return net(x.detach())
        else:
            return net(x).to(opt.device)

def mask(x, s, opt):
    if opt.inpainting:
        return opt.masks[s]*x
    else:
        return x