import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
     

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        image_height, image_width = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_height, self.patch_width = patch_size if isinstance(patch_size, tuple) else (
            patch_size, patch_size)

        assert image_height % self.patch_height == 0 and image_width % self.patch_width == 0, "Image dimensions must be divisible by the patch size."
        self.p1, self.p2 = (image_height // self.patch_height), (image_width // self.patch_width)
        self.num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)

        self.patch_embed = nn.Linear(in_chans * self.patch_height * self.patch_width, embed_dim)

    def forward(self, x):
        N, C, H, W = x.shape
        patches = x.reshape([N, C, self.p1, self.patch_height, self.p2, self.patch_width]).permute(
            [0, 2, 4, 1, 3, 5]).reshape([N, self.num_patches, -1])
        x = self.patch_embed(patches)
        x = x.flatten(2)
        return x, patches


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class MaskAutoencoderConv(nn.Module):
    def __init__(self, img_size=(256, 128), patch_size=(16, 8), in_chans=3, depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768], embed_dim=1024, drop_path_rate=0., layer_scale_init_value=1e-6,
                 in_planes=2048, mask_ratio=0.75):
        super(MaskAutoencoderConv, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # ============= encoding ===============================#
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # ============= encodor ===============================#

        # self.last_conv = nn.Conv2d(dims[-1], in_planes, kernel_size=1, stride=1, bias=False) # other
        
        self.last_conv = nn.Conv2d(dims[-1], in_planes, kernel_size=1, stride=1, padding=1, bias=False) # cuhk03

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.head.weight.data.mul_(self.head_init_scale)
        self.head.bias.data.mul_(self.head_init_scale)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def mask(self, x):
        inputs = x
        tokens, patches = self.patch_embed(x)

        batch, num_patches, _ = tokens.shape  # batch_size, num_patches, _

        num_masked = int(self.mask_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches).argsort(axis=-1).to(x.device)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        batch_range = torch.arange(batch)[:, None]
        image = patches.clone()  # 采样后的图

        mask = torch.ones([batch, num_patches], device=x.device)
        mask[:, :num_masked] = 0
        mask = torch.gather(mask, dim=1, index=rand_indices)

        image[batch_range, masked_indices] = 0  # mask sampling area
        img = self._reconstruct(image, self.img_size, self.patch_size)

        return img, inputs

    def _reconstruct(self, x, image_size, patch_size):
        """reconstrcunt [batch_size, num_patches, embedding] -> [batch_size, channels, h, w]"""
        B, N, _ = x.shape  # batch_size, num_patches, dim

        p1, p2 = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
        x = x.reshape([B, p1, p2, -1, patch_size[0], patch_size[1]]).permute([0, 3, 1, 4, 2, 5]).reshape(
            [B, -1, image_size[0], image_size[1]])
        return x

    def forward_econder(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # x = self.norm(x.mean([-2, -1]))
        return x

    def forward(self, x):
        mask_img, x = self.mask(x)
        fearture = self.forward_econder(x)
        fearture = self.last_conv(fearture)

        mask_fearture = self.forward_econder(mask_img)
        mask_fearture = self.last_conv(mask_fearture)
        return fearture, mask_fearture

    def load_param(self, model_path):
        print('Load weights from {}.'.format(model_path))
        param_dict = torch.load(model_path)["model"]

        for k in param_dict:
            # if "patch_embed." in k:
            #     continue
             
            if "decoder." in k:
                continue

            if "topsample." in k:
                continue

            if 'norm' in k:
                continue

            self.state_dict()[k].copy_(param_dict[k])



