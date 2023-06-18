import argparse
import glob
import h5py
import os
import cv2
import numpy as np
import PIL.Image as pil_image

def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    pd_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        hr = pil_image.open(image_path)
        hr = np.array(hr).astype(np.float32)
        for i in range(0, hr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, hr.shape[1] - args.patch_size + 1, args.stride):
                hr_patches.append(hr[i:i+args.patch_size, j:j+args.patch_size])
    hr_patches = np.array(hr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    for image_path in sorted(glob.glob('{}/*'.format(args.lr_dir))):
        lr = pil_image.open(image_path)
        lr = np.array(lr).astype(np.float32)
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i+args.patch_size, j:j+args.patch_size])
    lr_patches = np.array(lr_patches)
    h5_file.create_dataset('lr', data=lr_patches)

    for image_path in sorted(glob.glob('{}/*'.format(args.pd_dir))):
        pd = pil_image.open(image_path)
        pd = np.array(pd).astype(np.float32)
        for i in range(0, pd.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, pd.shape[1] - args.patch_size + 1, args.stride):
                pd_patches.append(pd[i:i+args.patch_size, j:j+args.patch_size])
    pd_patches = np.array(pd_patches)
    h5_file.create_dataset('pd', data=pd_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    pd_group = h5_file.create_group('pd')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path)
        hr = np.array(hr).astype(np.float32)
        hr_group.create_dataset(str(i), data=hr)

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.lr_dir)))):
        lr = pil_image.open(image_path)
        lr = np.array(lr).astype(np.float32)
        lr_group.create_dataset(str(i), data=lr)

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.pd_dir)))):
        # name = os.path.split(image_path)[-1]
        pd = pil_image.open(image_path)
        pd = np.array(pd).astype(np.float32)
        pd_group.create_dataset(str(i), data=pd)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='Datasets/NAMIC/train_hr')
    parser.add_argument('--lr-dir', type=str, default='Datasets/NAMIC/train_lr4')
    parser.add_argument('--ref-dir', type=str, default='Datasets/NAMIC/train_ref')
    parser.add_argument('--output-path', type=str, default='Datasets/NAMIC/train_randsample_4.h5')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=64)
    # parser.add_argument('--eval', action='store_false')
    args = parser.parse_args()
    # print(args.eval)
    # train(args)
    eval(args)
    # if not args.eval:
    #     train(args)
    # else:
    #     eval(args)
