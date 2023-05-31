
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_


class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()

        self.image_height, self.image_width = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_height, self.patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, \
            "Image dimensions must be divisible by patch size."

        self.p1, self.p2 = (self.image_height // self.patch_height), (self.image_width // self.patch_width)
        self.num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)

        self.patch_embed = nn.Linear(in_chans * self.patch_height * self.patch_width, embed_dim)

    def forward(self, x):
        N, C, H, W = x.shape
        patches = x.reshape([N, C, self.p1, self.patch_height, self.p2, self.patch_width]).permute(
            [0, 2, 4, 1, 3, 5]).reshape([N, self.num_patches, -1])

        x = self.patch_embed(patches)
        x = x.flatten(2)
        return x, patches


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.weigh = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.ones(normalized_shape))

        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weigh, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weigh[:, None, None] * x + self.bias[:, None, None]

            return x


class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, enOrde="en"):
        super(ConvBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim * 4)
        # self.act = nn.GELU()
        self.act = Mish()
        self.pwconv2 = nn.Linear(dim * 4, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.enOrde = enOrde

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)

        if self.enOrde == "en":
            x = self.drop_path(x)
            return x

        elif self.enOrde == "de":
            x = input + self.drop_path(x)
            return x


class Encoder(nn.Module):
    def __init__(self, patch_embed, in_chans, depths, dims, drop_path_rate, layer_scale_init_value, deploy, **kwargs):
        super(Encoder, self).__init__()
        self.patch_embed = patch_embed
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                nn.BatchNorm2d(dims[i + 1]),
                nn.ReLU(inplace=True)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value, enOrde="en") for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.conv = nn.ModuleList()
        # self.conv.append(nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1))

        for i in range(4):
            nc = nn.Conv2d(dims[i], dims[i], kernel_size=3, stride=1, padding=1)
            self.conv.append(nc)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

    def _reconstruct(self, x, image_size, patch_size):
        """reconstrcunt [batch_size, num_patches, embedding] -> [batch_size, channels, h, w]"""
        B, N, _ = x.shape  # batch_size, num_patches, dim

        p1, p2 = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
        x = x.reshape([B, p1, p2, -1, patch_size[0], patch_size[1]]).permute([0, 3, 1, 4, 2, 5]).reshape(
            [B, -1, image_size[0], image_size[1]])
        return x

    def mask(self, x, mask_ratio):
        tokens, patches = self.patch_embed(x)
        batch, num_patches, _ = tokens.shape  # batch_size, num_patches, _

        num_masked = int(mask_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches).argsort(axis=-1).to(x.device)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        batch_range = torch.arange(batch)[:, None]
        image = patches.clone()  # 采样后的图

        mask = torch.ones([batch, num_patches], device=x.device)
        mask[:, :num_masked] = 0
        mask = torch.gather(mask, dim=1, index=rand_indices)

        image[batch_range, masked_indices] = 0  # mask sampling area
        img = self._reconstruct(image, (self.patch_embed.image_height, self.patch_embed.image_width),
                                (self.patch_embed.patch_height, self.patch_embed.patch_width))
        return img, mask

    def forward(self, x, mask_ratio):
        x, mask = self.mask(x, mask_ratio=mask_ratio)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x_2 = self.conv[i](x)

            for j in range(self.depths[i]):
                x_3 = self.stages[i][j](x)
                x = x + x_2 + x_3

        x = self.norm(x.mean([-2, -1]))
        b, d = x.shape
        y = x.view(b, 1, 1, d).permute([0, 3, 1, 2])
        return y, mask


class Decoder(nn.Module):
    def __init__(self, in_chans, depths, dims, drop_path_rate=0., layer_scale_init_value=1e-6):
        super(Decoder, self).__init__()
        self.topsample_layers_decoder = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_decoder = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], dims[-1], kernel_size=(8, 4), padding=0),
            # LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.topsample_layers_decoder.append(stem_decoder)
        for i in range(3):
            topsample_layer_decoder = nn.Sequential(
                LayerNorm(dims[3 - i], eps=1e-6, data_format="channels_first"),
                nn.ConvTranspose2d(dims[3 - i], dims[2 - i], kernel_size=2, padding=0, stride=2),
            )
            self.topsample_layers_decoder.append(topsample_layer_decoder)

        self.stages_decoder = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_decoder = nn.Sequential(
                *[ConvBlock(dim=dims[3 - i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value, enOrde="de") for j in range(depths[3 - i])]
            )
            self.stages_decoder.append(stage_decoder)
            cur += depths[i]

        self.final_topsample = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(dims[0], dims[0], kernel_size=2, padding=0, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(dims[0], in_chans, kernel_size=2, padding=0, stride=2)
        )

    def forward(self, x):
        for i in range(4):
            x = self.topsample_layers_decoder[i](x)
            x = self.stages_decoder[i](x)

        x = self.final_topsample(x)
        return x


class MaskAutoencoderConv(nn.Module):
    def __init__(self, img_size=(256, 128), patch_size=(16, 8), in_chans=3, embed_dim=1024,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., layer_scale_init_value=1e-6):
        super(MaskAutoencoderConv, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.encoder = Encoder(patch_embed=self.patch_embed, in_chans=in_chans, depths=depths, dims=dims,
                               drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value,
                               deploy=True)
        self.decoder = Decoder(in_chans=in_chans, depths=depths, dims=dims, drop_path_rate=drop_path_rate,
                               layer_scale_init_value=layer_scale_init_value)

        self.initialize_weights()

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p_h = self.patch_embed.patch_height
        p_w = self.patch_embed.patch_width
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p_h
        w = imgs.shape[3] // p_w
        # x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))

        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p_h, w, p_w))
        # print("=======:{}".format(x.shape))
        x = torch.einsum('nchpwq->nhwpqc', x)
        # x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        x = x.reshape(shape=(imgs.shape[0], h * w, p_h * p_w * self.in_chans))
        return x

    def forward_loss(self, img, img_mask, mask):
        img_closs = self.patchify(img)

        img_mask_closs = self.patchify(img_mask)

        loss = (img_closs - img_mask_closs) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # print(loss.shape)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask_ratio):
        img_mask, mask = self.encoder(x, mask_ratio=mask_ratio)
        pred = self.decoder(img_mask)
        loss = self.forward_loss(x, pred, mask)

        if not self.training:
            return pred, img_mask

        return loss


def RPSNetV2_tiny(**kwargs):
    MAC = MaskAutoencoderConv(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    return MAC


tiny = RPSNetV2_tiny

# if __name__ == "__main__":
#     # img = cv2.imread("1.png")
#     # data = cv2.resize(img, dsize=(128, 256), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
#     # tensor_cv = torch.from_numpy(np.transpose(data, (2, 0, 1))).unsqueeze(0).float()
#
#     inputs = torch.rand([4, 3, 256, 128])
#     models = MaskAutoencoderConv()
#     out = models(inputs, mask_ratio=0.75)
#     print(out)
