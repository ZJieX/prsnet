from model.backbones import PRSNet
import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

import pretrain_utils.misc as misc
from model.backbones import PRSNet

parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
parser.add_argument('--model', default='tiny', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--device', default='cpu',
                    help='device to use for training / testing')
parser.add_argument('--weights', default="", help='Pretrain model')
parser.add_argument('--model_path', default="./pretrained_model/checkpoint-100.pth", help='model with trained')

parser.add_argument('--input_size', default=(256, 128), type=int,
                    help='images input size')
parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
parser.add_argument('--data_path', default='./pretrain_dataset/test/', type=str,
                    help='dataset path')
parser.add_argument('--seed', default=0, type=int)

args = parser.parse_args()


def mask(x, mask_ratio, img_size, patch_size, embed_dim, in_chans):
    image_height, image_width = img_size if isinstance(img_size, tuple) else (img_size, img_size)
    patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
    p1, p2 = (image_height // patch_height), (image_width // patch_width)
    # num_patches = (image_height // patch_height) * (image_width // patch_width)

    num_patches = (image_height // patch_height) * (image_width // patch_width)

    N, C, H, W = x.shape
    patches = x.reshape([N, C, p1, patch_height, p2, patch_width]).permute(
        [0, 2, 4, 1, 3, 5]).reshape([N, num_patches, -1])

    x = nn.Linear(in_chans * patch_height * patch_width, embed_dim)(patches)
    tokens = x.flatten(2)

    batch, num_patches, _ = tokens.shape

    num_masked = int(mask_ratio * num_patches)
    rand_indices = torch.rand(batch, num_patches).argsort(axis=-1)
    masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
    batch_range = torch.arange(batch)[:, None]
    image = patches.clone()  # 采样后的图

    mask = torch.ones([batch, num_patches], device=x.device)
    mask[:, :num_masked] = 0
    mask = torch.gather(mask, dim=1, index=rand_indices)

    image[batch_range, masked_indices] = 0  # mask sampling area
    img = reconstruct(image, img_size, patch_size)
    print(img.shape)
    return img, mask


def reconstruct(x, image_size, patch_size):
    """reconstrcunt [batch_size, num_patches, embedding] -> [batch_size, channels, h, w]"""
    B, N, _ = x.shape  # batch_size, num_patches, dim

    p1, p2 = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
    x = x.reshape([B, p1, p2, -1, patch_size[0], patch_size[1]]).permute([0, 3, 1, 4, 2, 5]).reshape(
        [B, -1, image_size[0], image_size[1]])
    return x


if __name__ == "__main__":
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(args.device)
    models = PRSNet.__dict__[args.model](weights=args.weights)
    models.to(device)

    print('Load weights from {}.'.format(args.model_path))
    weights_dict = torch.load(args.model_path, map_location=device)["model"]
    models.load_state_dict(weights_dict)

    models.eval()

    for f in os.listdir(args.data_path):
        img_path = os.path.join(args.data_path, f)
        img = cv2.imread(img_path)
        data = cv2.resize(img, dsize=(args.input_size[1], args.input_size[0]), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

        tensor_data = torch.from_numpy(np.transpose(data, (2, 0, 1))).unsqueeze(0).float()

        _range = np.max(data) - np.min(data)
        tensor_cv = (data - np.min(data)) / _range
        tensor_cv = torch.from_numpy(np.transpose(tensor_cv, (2, 0, 1))).unsqueeze(0).float()

        masked_img, _ = mask(tensor_data, mask_ratio=args.mask_ratio, img_size=(256, 128), patch_size=(16, 8),
                             embed_dim=1024, in_chans=3)

        masked_img = masked_img[0].numpy()
        masked_img = np.clip(masked_img, 0, 255).astype('uint8')
        masked_img = np.transpose(masked_img, [1, 2, 0])

        # tensor_cv = tensor_cv / 255.0
        pred, _ = models(tensor_cv, mask_ratio=args.mask_ratio)

        pred = pred[0].detach().numpy()
        pred = np.transpose(pred, [1, 2, 0])

        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

        plt.rcParams['figure.figsize'] = [24, 24]
        plt.subplot(131)
        plt.imshow(data)
        plt.subplot(132)
        plt.imshow(masked_img)
        plt.subplot(133)
        plt.imshow(pred)
        plt.title("mask ratio: 0.75", x=-0.7, y=1.05)
        plt.show()
        # cv2.imshow("orignal", data)
        # cv2.imshow("mask", masked_img)
        # cv2.imshow("result", pred)


