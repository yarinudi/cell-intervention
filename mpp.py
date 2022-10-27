import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pickle
import numpy as np

from tqdm import tqdm

from einops.layers.torch import Rearrange

import math
from einops import rearrange, repeat, reduce
"""                
    Modified version from the great work of MPP in vit-pytorch:
    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mpp.py
"""


def exists(val):
    return val is not None


def prob_mask_like(t, prob):
    """ Mask a patch with respect to the probabality to be masked. """
    batch, seq_length, _ = t.shape

    return torch.zeros((batch, seq_length)).float().uniform_(0, 1) < prob


def get_mask_subset_with_prob(patched_input, prob):
    """ Generates different mask for every image in the batch. """

    batch, seq_len, _, device = *patched_input.shape, patched_input.device
    masks = []
    for i in range(batch):    
        rand = int(torch.randint(1, seq_len, (1,), device=device))
        sampled_indices = torch.range(0, rand).type(torch.LongTensor).reshape((1, -1))
        new_mask = torch.zeros((1, seq_len), device=device)
        new_mask.scatter_(1, sampled_indices, 1)
        masks.append(new_mask)

    new_mask = torch.cat(masks, dim=0)

    return new_mask.bool()

def init_mpp(model, feats, model_args):
    mpp_trainer = MPP(
        transformer=model,
        patch_size=(1, 256),
        dim=model_args['dim'],
        device=model_args['device'],
        mask_prob=0.15,          # probability of using token in masked prediction task
        random_patch_prob=0,  # probability of randomly replacing a token being used for mpp
        replace_prob=1,       # probability of replacing a token being used for mpp with the mask token
        max_pixel_val=np.max(np.array(feats))
    )
    return mpp_trainer


def pretrain_mpp(feats, images_loader, model_args, epochs=50):

    mpp_trainer = init_mpp(feats, model_args)
    opt = torch.optim.Adam(mpp_trainer.parameters(), lr=3e-4)

    tot_loss = []

    for i in range(epochs):
        for images, _ in tqdm(images_loader):
            loss = mpp_trainer(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss.append(loss)

    tot_loss = [float(loss.detach().cpu()) for loss in tot_loss]
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 8))
    plt.plot(tot_loss, marker='o', linestyle='--', color='g')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('MPP Loss')

    # save your improved network
    torch.save(model.state_dict(), './pretrained-net.pt')

    return mpp_trainer, tot_loss


class MPPLoss(nn.Module):

    def __init__(
        self,
        patch_size,
        channels,
        output_channel_bits,
        max_pixel_val,
        mean,
        std
    ):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.output_channel_bits = output_channel_bits
        self.max_pixel_val = max_pixel_val

        self.mean = torch.tensor(mean).view(-1, 1, 1) if mean else None
        self.std = torch.tensor(std).view(-1, 1, 1) if std else None

    def forward(self, predicted_patches, target, mask):
        p, c, mpv, bits, device = self.patch_size, self.channels, self.max_pixel_val, self.output_channel_bits, target.device
        bin_size = mpv / (2 ** bits)

        # un-normalize input
        if exists(self.mean) and exists(self.std):
            target = target * self.std + self.mean

        # reshape target to patches
        target = target.clamp(max = mpv) # clamp just in case
        avg_target = reduce(target, 'b c (h p1) (w p2) -> b (h w) c', 'mean', p1 = p[0], p2 = p[1]).contiguous()

        channel_bins = torch.arange(bin_size, mpv, bin_size, device = device)
        discretized_target = torch.bucketize(avg_target, channel_bins)

        bin_mask = (2 ** bits) ** torch.arange(0, c, device = device).long()
        bin_mask = rearrange(bin_mask, 'c -> () () c')

        target_label = torch.sum(bin_mask * discretized_target, dim = -1)

        loss = F.cross_entropy(predicted_patches[mask], target_label[mask])
        return loss


class MPP(nn.Module):
    def __init__(
        self,
        transformer,
        patch_size,
        dim,
        device,
        output_channel_bits=1,
        channels=1,
        max_pixel_val=1.0,
        mask_prob=0.15,
        replace_prob=1,
        random_patch_prob=0.5,
        mean=None,
        std=None
    ):
        super().__init__()
        self.transformer = transformer
        self.loss = MPPLoss(patch_size, channels, output_channel_bits,
                            max_pixel_val, mean, std)

        # output transformation
        self.to_bits = nn.Linear(dim, 2**(output_channel_bits * channels))

        # vit related dimensions
        self.patch_size = patch_size

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob

        # token ids
        self.mask_token = nn.Parameter(torch.randn(1, 1, channels * patch_size[0] *patch_size[1]))

        self.device = device

    def mask_input(self, input, **kwargs):
        # clone original image for loss
        img = input.clone().detach()

        # reshape raw image to patches
        p = self.patch_size
        input = rearrange(input,
                          'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=p[0],
                          p2=p[1])

        mask = get_mask_subset_with_prob(input, self.mask_prob)
    
        # mask input with mask patches with probability of `replace_prob` (keep patches the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()
    
        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob).to(mask.device)
        bool_mask_replace = ((mask * replace_prob) == False)
    
        masked_input[bool_mask_replace] = self.mask_token        
        masked_input = masked_input.reshape((masked_input.shape[0], 1, masked_input.shape[1], -1))

        return masked_input, img, mask

    def forward(self, input, **kwargs):

        transformer = self.transformer
        # clone original image for loss
        img = input.clone().detach()

        # reshape raw image to patches
        p = self.patch_size
        input = rearrange(input,
                          'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=p[0],
                          p2=p[1])

        mask = get_mask_subset_with_prob(input, self.mask_prob)

        # mask input with mask patches with probability of `replace_prob` (keep patches the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob).to(mask.device)
        bool_mask_replace = ((mask * replace_prob) == True)
        
        masked_input[bool_mask_replace] = self.mask_token

        # linear embedding of patches
        masked_input = transformer.to_patch_embedding[-1](masked_input)

        # add positional embeddings to input
        masked_input = masked_input + transformer.positional_encoder(masked_input)
        masked_input = transformer.dropout(masked_input)

        # get generator output and get mpp loss
        masked_input = transformer.transformer(masked_input, **kwargs)

        logits = self.to_bits(masked_input)

        mpp_loss = self.loss(logits, img, mask)

        return mpp_loss
