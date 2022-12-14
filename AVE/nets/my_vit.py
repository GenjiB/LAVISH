import timm
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from ipdb import set_trace
import torch.nn.functional as F

from timm.models.layers import to_2tuple,trunc_normal_

# override the timm package to relax the input shape constraint.
# class PatchEmbed(nn.Module):
#     def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=1024):
#         super().__init__()

#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         # Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10)) for audioset
#     def forward(self, x):
#         # x = 1x1x128x1024
#         x = self.proj(x)
#         patch_info_4d = x.shape
#         x = x.flatten(2).transpose(1, 2) # 768x12x101
#         return x, patch_info_4d

class my_PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=1024, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x,is_shape_info=False):
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        patch_info_4d = x.shape
        x = x.flatten(2).transpose(1, 2) # 768x12x101
        # x = self.norm(x)
        if is_shape_info:
            return x, patch_info_4d
        else:
            return x


class my_vit(nn.Module):
    """
    """
    def __init__(self, name=''):
        
        super(my_vit, self).__init__()

        # override timm input shape restriction (v0.4.5)
        # timm.models.vision_transformer.PatchEmbed = PatchEmbed #(img_size=224, patch_size=32, in_chans=3, embed_dim=1024)
        # timm.models.layers.patch_embed.PatchEmbed = PatchEmbed(img_size=224, patch_size=32, in_chans=3, embed_dim=1024)
        # timm.models.layers.patch_embed.PatchEmbed = my_PatchEmbed(img_size=224, patch_size=32, in_chans=3, embed_dim=1024)

        # timm.models.vision_transformer.patch_embed
        # timm.models.vision_transformer.VisionTransformer.patch_embed = my_PatchEmbed

        

        self.v = timm.create_model(name, pretrained=True)

        # self.ViT.v.patch_embed.proj.weight



        ### -------> yb: custom forward (v0.6.7)
        my_conv = my_PatchEmbed(img_size=self.v.patch_embed.img_size[0], patch_size=self.v.patch_embed.patch_size[0], in_chans=3, embed_dim=self.v.embed_dim)
        my_conv.proj.load_state_dict(self.v.patch_embed.proj.state_dict(), strict=True)
        self.v.patch_embed = my_conv
        ###### <-------


        # #### --------------> yb: adapt MAE weights. should only skip "head.weight", "head.bias"
        # self.v.load_state_dict(torch.load('./ckpt/mae_pretrain_vit_large.pth')['model'], strict=False)
        # ### <------




        

    def forward_patch(self,x, is_shape_info=False):
        x, patch_info_4d = self.v.patch_embed(x, is_shape_info=is_shape_info)

        ### deit ------>
        # x = torch.cat((
        #     self.v.cls_token.expand(x.shape[0], -1, -1),
        #     self.v.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        ### <-------

        ### standard vit -------->
        x = torch.cat((
            self.v.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        ### <---------


        # adapt visual to audio spec
        if self.v.pos_embed.size(1) != x.size(1):
            x = self.v.pos_drop(x + F.interpolate(self.v.pos_embed.permute(0,2,1), x.size(1), mode='linear').permute(0,2,1))
        else:
            x = self.v.pos_drop(x + self.v.pos_embed)

        if is_shape_info:
            return x, patch_info_4d
        else:
            return x

    @autocast()
    def forward_features(self, x, additional_patch=None) -> torch.Tensor:
        x = self.v.patch_embed(x)


        # ### deit ------>
        # x = torch.cat((
        #     self.v.cls_token.expand(x.shape[0], -1, -1),
        #     self.v.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # ### <-------

        
        
        
        ## standard vit -------->
        x = torch.cat((
            self.v.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        ## <---------

        # -----------> adapt visual to audio spec
        if self.v.pos_embed.size(1) != x.size(1):
            x = self.v.pos_drop(x + F.interpolate(self.v.pos_embed.permute(0,2,1), x.size(1), mode='linear').permute(0,2,1))
        else:
            x = self.v.pos_drop(x + self.v.pos_embed)
        # ######### <---------
        


        if additional_patch is not None:
            x = torch.cat((x,additional_patch), dim=1) 


        # vis tokens: 578 ; audio tokens: 110 
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        if pre_logits:
            return (x[:, 0] + x[:, 1]) / 2
        x, x_dist = self.v.head(x[:, 0]), self.v.head_dist(x[:, 1])
        if self.v.distilled_training and self.v.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2
    @autocast()
    def forward(self, x):
        x = self.v.forward_features(x)
        x = self.v.forward_head(x)
        return x