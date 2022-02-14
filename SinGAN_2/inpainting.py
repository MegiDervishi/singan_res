from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions
import torch
import cv2

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--inpainting_start_scale', help='inpainting injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='inpainting')
    #parser.add_argument('--radius', help='radius harmonization', type=int, default = 10)
    #parser.add_argument('--ref_name', help='training image name', type = str, default = "")
    parser.add_argument('--mask_dir', help='mask inpaint dir', default='Input/Masks')
    parser.add_argument('--mask_name', help='mask image name', default='')
    parser.add_argument('--patch_fill', help='choose patch fill mode', default='inpaint')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    if opt.mask_name =="":
        opt.mask_name = opt.input_name
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.inpainting_start_scale < 1) | (opt.inpainting_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            #Average color
            mask_name, suffix = opt.mask_name.split('.')
            m = cv2.imread(f'{opt.mask_dir}/{mask_name}_mask.{suffix}')
            img = cv2.imread(f'{opt.input_dir}/{opt.input_name}')


            if opt.patch_fill == 'inpaint':
                gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                img = cv2.inpaint(img,gray,3,cv2.INPAINT_TELEA)
            elif opt.patch_fill == 'average':
                m = 1 - m / 255
                for j in range(3):
                    img[:, :, j][m[:, :, j] < 0.5] = img[:,:, j][m[:,:, j] > 0.5].mean()
            elif opt.patch_fill == 'neigh_average':
                m = 1 - m / 255
                mask = functions.read_image_dir(f"{opt.mask_dir}/{mask_name}_mask.{suffix}", opt)
                dilated_mask = functions.dilate_mask(mask, opt)
                dilated_mask = functions.torch2uint8(dilated_mask).astype(np.float64)
                dilated_mask = 1 - dilated_mask / 255
                for j in range(3):
                    img[:, :, j][m[:, :, j] < 0.5] = np.mean(img[dilated_mask[:, :, j] * (1 - m[:, :, j]) < np.max(dilated_mask)/2])
            elif opt.patch_fill == 'none':
                pass
            else:
                raise RuntimeError("Unknown patch filling option, must be in ['inpaint', 'average', 'neigh_average', 'none']")

            # save the patch_filled image
            img_name, suffix = opt.input_name.split('.')
            cv2.imwrite(f"{opt.input_dir}/{img_name}_filled.{suffix}", img)

            # From here on we don't need to change anything, simply use SinGan to harmonize 
            # using the usual SinGan code.
            ref = functions.read_image_dir(f"{opt.input_dir}/{img_name}_filled.{suffix}", opt)
            mask_name, suffix = opt.mask_name.split('.')
            mask = functions.read_image_dir(f"{opt.mask_dir}/{mask_name}_mask.{suffix}", opt)

            if ref.shape[3] != real.shape[3]:
                mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]

            mask = functions.dilate_mask(mask, opt)

            N = len(reals) - 1
            n = opt.inpainting_start_scale
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            out = (1-mask)*real+mask*out
            plt.imsave('%s/start_scale=%d.png' % (dir2save,opt.inpainting_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)