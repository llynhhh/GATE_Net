import argparse
import torch
import  os
import numpy as np
import PIL.Image as pil_image
import pytorch_ssim
import time
# from thop import profile
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import pandas as pd
# from uformer import Uformer
from model_3_10_1_ import Model
# from model_stage1_block1_k5_cc0_encoder1 import Model
from datasets import TestDataset
from utils import AverageMeter, calc_psnr
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights-file', type=str, default='outputs_NAMIC/random_sample_stage1_fblock3_cc2_nomfsa/x4/best.pth')
    # parser.add_argument('--weights-file', type=str, default='50/proposed/3_10_1/dataset1/radial_namic/x4/best.pth')
    parser.add_argument('--weights-file', type=str, default='results/NAMIC/x4/best.pth')
    # parser.add_argument('--weights-file', type=str, default='50/uformer/dataset1/x4/best.pth')
    parser.add_argument('--test-file', type=str, default='/home/amax/Documents/Datasets/NAMIC/test_randsample_4.h5')
    # parser.add_argument('--test-file', type=str, default='/home/ynl/Documents/Datasets/2-MICCAI_BraTS_2018/test_lr4.h5')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    torch.cuda.empty_cache()
    cudnn.benchmark = True
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = Model().to(device)
    # model = Uformer(img_size=256, in_chans=2, embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
    #                 win_size=8, mlp_ratio=4., qkv_bias=True,
    #                 token_projection='linear', token_mlp='leff', se_layer=False)
    # model = model.to(device)
    # model = Model()
    # model = torch.nn.DataParallel(model)
    # model.cuda()
    # path = '/home/ynl/Documents/Datasets/mask_4.xlsx'
    # mask = pd.read_excel(path)
    # mask = mask.values
    # mask = torch.from_numpy(mask)
    # mask = mask.to(device)
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))

    test_dataset = TestDataset(args.test_file)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)


    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    model.eval()

    psnr = 0
    ssim = 0
    psnr_max = 0
    ssim_max = 0
    psnr_min = 45
    ssim_min = 1
    PSNR = []
    SSIM = []
    start = time.time()
    for i, data in enumerate(test_dataloader):
        inputs, pd, labels = data
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()
        pd = pd.to(device).float()
        # inputs = torch.cat((inputs, pd), 1)
        with torch.no_grad():
            preds = model(inputs, pd).clamp(0.0, 1.0)
        psnr_ = calc_psnr(preds, labels)
        if psnr_ > psnr_max:
            psnr_max = psnr_
        if psnr_ < psnr_min:
            psnr_min = psnr_
        ssim_ = pytorch_ssim.ssim(preds, labels)
        if ssim_ > ssim_max:
            ssim_max = ssim_
        if ssim_ < ssim_min:
            ssim_min = ssim_
        psnr += psnr_
        ssim += ssim_
        PSNR.append(psnr_)
        SSIM.append(ssim_)


        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        output = np.array(preds)
        output = output.astype(np.uint8)
        output = pil_image.fromarray(output)
        # plt.imshow(output)
        # plt.show()
        output.save(os.path.join('output/NAMIC/x4', '{}.png'.format(str(i))))
    end = time.time()
    dis = end - start
    psnr = psnr / len(test_dataset)
    ssim = ssim / len(test_dataset)
    # np.save('/home/ynl/Desktop/PSNR4.npy', PSNR)
    # np.save('/home/ynl/Desktop/SSIM4.npy', SSIM)

    psnr_std = torch.std(torch.stack(PSNR))
    ssim_std = torch.std(torch.stack(SSIM))

    print('eval psnr: {:.2f}    max psnr: {:.2f}   min psnr: {:.2f}'.format(psnr, psnr_max, psnr_min))
    print('eval ssim: {:.4f}    max ssim: {:.4f}   min ssim: {:.4f}'.format(ssim, ssim_max, ssim_min))
    print('psnr_std: ', psnr_std)
    print('ssim_std: ', ssim_std)
    print('dis: ', dis)



