#!/usr/bin/env python3
# encoding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn import Linear, Dropout
from model.KamxT import kMaXTransformerLayer
from model.kmax_pixel_decoder import ConvBN, get_activation


class Mlp(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, dropout_rate=0.1):
        super(Mlp, self).__init__()
        self.fc1 = Linear(dim_input, dim_hidden)
        self.fc2 = Linear(dim_hidden, dim_output)
        self.act_fn = get_activation('relu')
        self.dropout = Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x


class encoder(nn.Module):
    def __init__(self, init_channels=8):
        super(encoder, self).__init__()
        self.init_channels = init_channels
        self.conv1a = nn.Conv3d(1, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels * 1)  # 32

        self.ds1 = torch.nn.MaxPool3d(2)

        self.conv2b = BasicBlock(init_channels * 1, init_channels * 2)

        self.ds2 = torch.nn.MaxPool3d(2)

        self.conv3b = BasicBlock(init_channels * 2, init_channels * 4)

        self.ds3 = torch.nn.MaxPool3d(2)

        self.conv4b = BasicBlock(init_channels * 4, init_channels * 8)

    def forward(self, x):
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)
        c2 = self.conv2b(c1d)
        c2d = self.ds2(c2)
        c3 = self.conv3b(c2d)
        c3d = self.ds3(c3)
        c4 = self.conv4b(c3d)
        return [c1, c2, c3, c4]


class decoder(nn.Module):
    def __init__(self, init_channels=8):
        super(decoder, self).__init__()
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)
        self.up4convb = BasicBlock(init_channels * 8, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 4, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels * 2, init_channels)

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.up1conv = nn.Conv3d(init_channels, 3, (1, 1, 1))

    def forward(self, x):
        for l in x:
            print(l.shape)
        u4 = self.up4conva(x[3])
        u4 = self.up4(u4)
        u4 = torch.cat([u4, x[2]], 1)
        u4 = self.up4convb(u4)

        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = torch.cat([u3, x[1]], 1)
        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = torch.cat([u2, x[0]], 1)
        u2 = self.up2convb(u2)

        uout = self.up1conv(u2)

        return uout


class UNet3D_g(nn.Module):
    """
    A normal 3D - Unet, different from the original model architecture in Ref, which use the content and style to reconstruc the high level feature.
    
    3d unet
    Ref:
        3D MRI brain tumor segmentation using autoencoder regularization. Andriy Myronenko
    Args:
        input_shape: tuple, (height, width, depth)
    """

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2, CC_modalities=4,
                 CC_classes=3, deep_supervised=False, pre_train=True, pretrainTest=False):
        super(UNet3D_g, self).__init__()
        self.CC_modalities = CC_modalities
        self.CC_classes = CC_classes
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.pre_train = pre_train
        self.ds = deep_supervised
        self.pretrainTest = pretrainTest
        self.make_encoder()
        self.make_KmaxT()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)
        self.row = Mlp(self.CC_classes * self.CC_modalities, 128, self.CC_modalities)
        self.col = Mlp(self.CC_classes * self.CC_modalities, 128, self.CC_classes)
        self.col64 = Mlp(self.CC_classes * self.CC_modalities, 128, self.CC_classes)
        self.col32 = Mlp(self.CC_classes * self.CC_modalities, 128, self.CC_classes)

    def make_KmaxT(self):
        self.layers = 6
        self.Q_channels = 128
        K_channels = 128
        V_channels = 128
        dropPathProb = 0.2
        self._cluster_centers = nn.Embedding(self.Q_channels, self.CC_modalities * self.CC_classes)
        self._kmax_transformer_layers = nn.ModuleList()
        for i in range(self.layers):
            self._kmax_transformer_layers.append(
                kMaXTransformerLayer(
                    num_classes=self.CC_classes,
                    in_channel_K=self.Q_channels,
                    in_channel_V=self.Q_channels,
                    in_channel_query=self.Q_channels,
                    nums_q=self.CC_modalities * self.CC_classes,
                    base_filters=128,
                    num_heads=8,
                    bottleneck_expansion=2,
                    key_expansion=1,
                    value_expansion=2,
                    drop_path_prob=dropPathProb)
            )

        self._128_16 = ConvBN(128, 16, kernel_size=1, bias=False,
                              norm='1b', act='gelu', conv_type='1d')
        self._128_32 = ConvBN(128, 32, kernel_size=1, bias=False,
                              norm='1b', act='gelu', conv_type='1d')
        self._128_64 = ConvBN(128, 64, kernel_size=1, bias=False,
                              norm='1b', act='gelu', conv_type='1d')

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1a = nn.Conv3d(self.in_channels, init_channels, (3, 3, 3), padding=(1, 1, 1))
        self.conv1b = BasicBlock(init_channels, init_channels)  # 32

        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, (3, 3, 3), stride=(2, 2, 2),
                             padding=(1, 1, 1))  # down sampling and add channels

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)

        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 4, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)

        self.ds3 = nn.Conv3d(init_channels * 4, init_channels * 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)

        self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, (1, 1, 1))
        self.up4 = nn.Upsample(scale_factor=2)  # mode='bilinear'
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, (1, 1, 1))
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, (1, 1, 1))
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.head_seg = nn.Conv3d(self.CC_classes * self.CC_modalities, 3, (1, 1, 1))
        self.head_reg = nn.Conv3d(init_channels * self.CC_modalities, 4, (1, 1, 1))
        self.ds_out = []
        self.up_out = []
        if self.ds:
            self.ds_out.append(nn.Conv3d(self.CC_classes * self.CC_modalities, self.out_channels, (1, 1, 1)))
            self.up_out.append(nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True))

            self.ds_out.append(nn.Conv3d(self.CC_classes * self.CC_modalities, self.out_channels, (1, 1, 1)))
            self.up_out.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))

            self.ds_out = nn.ModuleList(self.ds_out)

    def forward(self, x):
        c1 = self.conv1a(x)  # B 16 128 128 128
        c1 = self.conv1b(c1)  # B 16 128 128 128
        c1d = self.ds1(c1)  # B 32 64 64 64

        c2 = self.conv2a(c1d)  # B 32 64 64 64
        c2 = self.conv2b(c2)  # B 32 64 64 64
        c2d = self.ds2(c2)  # B 64 32 32 32
        c2d_p = self.pool(c2d)  # B 64 16 16 16

        c3 = self.conv3a(c2d)  # B 64 32 32 32
        c3 = self.conv3b(c3)  # B 64 32 32 32
        c3d = self.ds3(c3)  # B 128 16 16 16

        c4 = self.conv4a(c3d)  # B 128 16 16 16
        c4 = self.conv4b(c4)  # B 128 16 16 16
        c4 = self.conv4c(c4)  # B 128 16 16 16
        c4d = self.conv4d(c4)  # B 128 16 16 16

        style = [c2d, c3d, c4d]  # [B 64 32 32 32] [B 128 16 16 16] [B 128 16 16 16]
        content = c4d  # B 128 16 16 16
        B, C, H, W, D = content.size()  # B 128 16 16 16
        cluster_centers = self._cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1)  # B x C x L [B 128 cluster_centers]
        prediction_result = content  # [B 128 16 16 16]
        for i in range(self.layers):
            cluster_centers, prediction_result = self._kmax_transformer_layers[i](
                pixel_featureK=prediction_result,
                pixel_featureV=prediction_result,
                query_feature=cluster_centers
            )
        u4 = self.up4conva(prediction_result)  # B 128 16 16 16  c4d->prediction_result

        u4 = self.up4(u4)  # B 64 32 32 32
        u4 = u4 + c3  # B 64 32 32 32
        u4 = self.up4convb(u4)  # B 64 32 32 32
        u3 = self.up3conva(u4)  # B 32 32 32 32
        u3 = self.up3(u3)  # B 32 64 64 64
        u3 = u3 + c2  # B 32 64 64 64
        u3 = self.up3convb(u3)  # B 32 64 64 64

        u2 = self.up2conva(u3)  # B 16 64 64 64
        u2 = self.up2(u2)  # B 16 128 128 128
        u2 = u2 + c1  # B 16 128 128 128
        u2 = self.up2convb(u2)  # B 16 128 128 128

        cluster_centers16 = self._128_16(cluster_centers)
        CC_modal = self.row(cluster_centers16)  # B 16 modalities
        CC_class = self.col(cluster_centers16)

        uout_classify = torch.einsum('bnc,bcxyz->bnxyz', F.normalize(CC_class, p=2, dim=1).transpose(1, 2),
                                     F.normalize(u2, p=2, dim=1))  # B CC_class->3 128 128 128

        if self.pre_train and not self.pretrainTest:
            u2reconstruction = torch.einsum('bcn,bcxyz->bcnxyz', CC_modal.float(), u2.float()) \
                .view(B, -1, 128, 128, 128)
            uout_reconstruction = self.head_reg(u2reconstruction)  # B 16->4 128 128 128
            uout = {'uout_reconstruction': uout_reconstruction, 'uout_classify': uout_classify}
        elif self.pretrainTest:
            u2reconstruction = torch.einsum('bnc,bcxyz->bcnxyz', CC_modal.float().transpose(1, 2), u2.float()) \
                .view(B, -1, 128, 128, 128)
            uout_reconstruction = self.head_reg(u2reconstruction)  # B 16->4 128 128 128
            uout = uout_reconstruction
        else:
            uout = uout_classify

        if self.ds and self.training:
            cluster_centers64 = self._128_64(cluster_centers)
            CC64_class = self.col64(cluster_centers64)
            cluster_centers32 = self._128_32(cluster_centers)
            CC32_class = self.col32(cluster_centers32)
            out4 = torch.einsum('bnc,bcxyz->bnxyz', F.normalize(CC64_class, p=2, dim=1).transpose(1, 2),
                                F.normalize(self.up_out[0](u4), p=2, dim=1))  # 分类
            out3 = torch.einsum('bnc,bcxyz->bnxyz', F.normalize(CC32_class, p=2, dim=1).transpose(1, 2),
                                F.normalize(self.up_out[1](u3), p=2, dim=1))  # 分类
            uout = [out4, out3, uout_classify]

        return uout, style, content


def ShuffleIndex_with_MDP(index: list, sample_ratio: float, mdp=0, mask=True):
    temp_index = index.copy()
    mdp_list = []
    mdp = np.random.randint(0, mdp + 1)

    if mdp > 3:
        mdp = 3
    for l in range(mdp):

        cindex = np.random.randint(0, 4)
        while cindex in mdp_list:
            cindex = np.random.randint(0, 4)
        mdp_list.append(cindex)

    if len(mdp_list) != 0:
        for l in mdp_list:
            for ls in range(l * 512, (l + 1) * 512):
                temp_index.remove(ls)

    sample_list = []
    if len(index) < 4:
        raise ValueError("ipnuts must be more than 4")
    elif mask:
        sample_length = int((1 - sample_ratio) * len(index))

        while len(sample_list) < sample_length:
            sample = random.choice(temp_index)
            sample_list.append(sample)
            temp_index.remove(sample)

        mask_list = [x for x in index if
                     x not in sample_list]  # get the remain index not in cls token and not in sample_index

    else:
        # only with MDP
        sample_list = temp_index
        mask_list = [x for x in index if x not in sample_list]

    return sample_list, mask_list


def ShuffleIndex_with_mask_modal(index: list, mask_modal=[], patch_shape=128):
    interal = int(np.power(patch_shape / 16, 3))
    temp_index = index.copy()

    mdp_list = mask_modal

    if len(mdp_list) != 0:
        for l in mdp_list:
            for ls in range(l * interal, (l + 1) * interal):
                temp_index.remove(ls)
    # print(mdp, len(temp_index))
    sample_list = []

    sample_list = temp_index
    mask_list = [x for x in index if x not in sample_list]

    return sample_list, mask_list


def proj(image, patch_size=16):
    B, C, D, H, W = image.shape
    image_ = image.reshape(B, C, D // patch_size, patch_size, H // patch_size, patch_size, W // patch_size, patch_size)

    image_ = image_.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(B, C * D // patch_size * H // patch_size * W // patch_size,
                                                            patch_size, patch_size, patch_size)

    return image_


def MaskEmbeeding2(B, mask_ratio=0.75, raw_input=None, patch_size=16, mdp=0, mask=True, mask_modal=[], patch_shape=128):
    """get the mask embeeding after patch_emb + pos_emb
    """

    D, H, W = patch_shape, patch_shape, patch_shape
    token_index = [x for x in range(0, int(np.power(patch_shape / 16, 3)) * 4)]

    if len(mask_modal) == 0:  # do not mask specific modal
        sample_index, mask_index = ShuffleIndex_with_MDP(token_index, mask_ratio, mdp=mdp, mask=mask)
    else:
        if -1 in mask_modal:
            sample_index, mask_index = ShuffleIndex_with_mask_modal(token_index, mask_modal=[], patch_shape=patch_shape)
        else:
            sample_index, mask_index = ShuffleIndex_with_mask_modal(token_index, mask_modal=mask_modal,
                                                                    patch_shape=patch_shape)

    decoder_embeeding = torch.zeros((B, raw_input.shape[1], patch_size, patch_size, patch_size)).to(raw_input.device)
    decoder_embeeding[:, sample_index, :, :, :] = raw_input[:, sample_index, :, :, :]
    decoder_embeeding = decoder_embeeding.reshape(B, 4, D // patch_size, H // patch_size, W // patch_size, patch_size,
                                                  patch_size, patch_size).permute(0, 1, 2, 5, 3, 6, 4, 7)

    decoder_embeeding = decoder_embeeding.reshape(B, 4, D, H, W)

    return decoder_embeeding


def random_block_mask_3d_v2(tensor, mask_ratio=0.15, min_block_size=4, max_block_size=16, mask_value=0, cong=0):
    if tensor.dim() == 5:
        spatial_dims = tensor.shape[2:]
        batch_sizes = tensor.shape[0]
        modales = tensor.shape[1]
    else:
        return 0

    mask = torch.ones_like(tensor, device=tensor.device, dtype=int)
    for B in range(batch_sizes):
        for modal in range(modales):

            total_pixels = mask.numel()
            target_masked_pixels = int(total_pixels * mask_ratio)

            current_masked_pixels = 0

            while current_masked_pixels < target_masked_pixels:
                d = random.randint(0, spatial_dims[0] - min_block_size)
                h = random.randint(0, spatial_dims[1] - min_block_size)
                w = random.randint(0, spatial_dims[2] - min_block_size)

                block_d = random.randint(min_block_size, min(max_block_size, spatial_dims[0] - d))
                block_h = random.randint(min_block_size, min(max_block_size, spatial_dims[1] - h))
                block_w = random.randint(min_block_size, min(max_block_size, spatial_dims[2] - w))

                block_pixels = block_d * block_h * block_w

                if current_masked_pixels + block_pixels > target_masked_pixels * 1.1:  # 允许10%的溢出
                    continue

                mask[B, modal, d:d + block_d, h:h + block_h, w:w + block_w] = 0
                current_masked_pixels += block_pixels

        mask3 = torch.ones([tensor.shape[0], tensor.shape[1] - 1, tensor.shape[2], tensor.shape[3], tensor.shape[4]],
                           device=tensor.device, dtype=int)
        mask1 = torch.zeros([tensor.shape[0], 1, tensor.shape[2], tensor.shape[3], tensor.shape[4]],
                            device=tensor.device, dtype=int)
        mask4 = torch.cat([mask3, mask1], dim=1)
        masked_tensor = tensor * mask * mask4 + mask_value * (1 - mask)
    return masked_tensor, mask * mask4


class Unet_missing(nn.Module):
    def __init__(self, input_shape, in_channels=4, out_channels=4, init_channels=16, p=0.2, pre_train=False,
                 deep_supervised=False, mdp=0, mask_modal=[], patch_shape=128, mask_ratio=0.875, augment=False,
                 CC_modalities=4, CC_classes=4, pretrainTest=False):
        super(Unet_missing, self).__init__()
        self.unet = UNet3D_g(input_shape, in_channels, out_channels, init_channels, p, pre_train=pre_train,
                             deep_supervised=deep_supervised, CC_modalities=CC_modalities, CC_classes=CC_classes,
                             pretrainTest=pretrainTest)

        self.limage = nn.Parameter(torch.zeros((1, 4, 155, 240, 240)), requires_grad=False)

        self.patch_shape = patch_shape
        self.raw_input = proj(torch.ones((1, 4, patch_shape, patch_shape, patch_shape)))
        self.token_index = [x for x in range(0, self.raw_input.shape[1])]
        self.pre_train = pre_train
        self.mask_ratio = mask_ratio
        self.mdp = mdp
        self.mask_modal = mask_modal
        self.augment = augment

    def forward(self, x, location=None, fmdp=None, aug_choises=None):
        if self.pre_train and location == None:  # never into this branch
            mask = MaskEmbeeding2(x.shape[0], mask_ratio=self.mask_ratio, raw_input=self.raw_input.to(x.device),
                                  mdp=self.mdp, mask=True)
            x = x * mask + self.limage[:, :, :128, :128, :128] * (1 - mask)
        elif self.pre_train and self.training:  # pretrain: patch mask
            if x.shape == 1:
                mask = MaskEmbeeding2(1, mask_ratio=self.mask_ratio, raw_input=self.raw_input.to(x.device),
                                      mdp=self.mdp, mask=True)
            else:
                for l in range(x.shape[0]):
                    mask = MaskEmbeeding2(1, mask_ratio=self.mask_ratio, raw_input=self.raw_input.to(x.device),
                                          mdp=self.mdp, mask=True)
                    x = x * mask + self.limage[:, :, :128, :128, :128] * (1 - mask)

        elif self.mdp != 0 and self.training:
            if fmdp == None:
                mask = torch.ones_like(x)
                pass

            else:
                if fmdp == 0:
                    cmask_modal = [-1]
                else:
                    cmask_modal = []
                    cmdp = fmdp.cpu()[0]
                    for l in range(cmdp):

                        cindex = np.random.randint(0, 4)
                        while cindex in cmask_modal:
                            cindex = np.random.randint(0, 4)
                        cmask_modal.append(cindex)
                mask = MaskEmbeeding2(x.shape[0], raw_input=self.raw_input.to(x.device), mdp=self.mdp, mask=False,
                                      mask_modal=cmask_modal)
                x = x * mask + self.limage[:, :, location[0][0]: location[0][1], location[1][0]: location[1][1],
                               location[2][0]: location[2][1]] * (1 - mask)  # detach here
        else:
            mask = torch.ones_like(x)

        if len(self.mask_modal) != 0 and not self.training:  # in inference
            mask = MaskEmbeeding2(x.shape[0], mask_ratio=self.mask_ratio, raw_input=self.raw_input.to(x.device),
                                  mdp=self.mdp, mask=False, mask_modal=self.mask_modal, patch_shape=self.patch_shape)
            x = x * mask + self.limage[:, :, location[0][0]: location[0][1], location[1][0]: location[1][1],
                           location[2][0]: location[2][1]].detach() * (1 - mask)
        uout, style, content = self.unet(x)
        if self.training:
            return uout, mask.sum((2, 3, 4)), style, content
        else:
            return uout
