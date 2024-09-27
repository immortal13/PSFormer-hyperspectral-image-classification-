import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from einops import rearrange, repeat
import math

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) ### 这里有没有dropout
        )
    def forward(self, x):
        return self.net(x)

## Spectral Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention, self).__init__()
        inner_dim = 256
        self.num_heads = num_heads
        self.temperature = dim ** -0.5 #nn.Parameter(torch.ones(num_heads, 1, 1))

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads #_为多少维的embedding，n是有多少个embedding，h = 3
        qkv = self.to_qkv(x).chunk(3, dim = -1) # b, n, inner_dim * 3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b h d n -> b n (h d)') # b, n, h*inner_dim 把8个头concate成一个头
        out =  self.to_out(out)
        return out

class Encoder_layer(nn.Module):
    def __init__(self, dim, num_head=1, mlp_ratio=1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_head)
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio))
 
    def forward(self, x): #[L,C]
        x = x.unsqueeze(0)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x[0]

class bn_1dconv_lrelu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(bn_1dconv_lrelu, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.act = nn.LeakyReLU(inplace=True)
        self.linear = nn.Sequential(nn.Linear(input_dim, output_dim)) ## same with nn.Conv1d
    def forward(self, H): #[L,C]
        H = self.bn(H)
        out = self.linear(H)
        out = self.act(out)
        return out

def bn_bsconv_lrelu(in_c, out_c, kernel_size):
    return nn.Sequential(
        nn.BatchNorm2d(in_c),
        nn.Conv2d(in_c, out_c, 1, padding=0, bias=False),
        nn.Conv2d(out_c, out_c, kernel_size=kernel_size,
            stride=1, padding=kernel_size//2, groups=out_c),
        nn.LeakyReLU()
    )

class Gate(nn.Module):
    def __init__(self, in_size):
        super(Gate, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, 1),
            nn.Sigmoid(),
        )
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (z+(beta * z)).sum(1), beta

# psformer.PSFormer(device, height, width, bands, class_count, S_list_gpu, stage_num, encoder_num)
class PSFormer(nn.Module):
    def __init__(self, channel, class_count, S_list, stage_num, encoder_num):
        super(PSFormer, self).__init__()

        num_layer = [encoder_num]*5
        layer_channels = 128
        layers_per_depth = [layer_channels]
        for i in range(stage_num):
            layer_channels = layer_channels//2
            layers_per_depth.append(layer_channels) ## [128, 64, 32, 16, 8]

        self.stage_num = stage_num    
        self.S_list = S_list
        self.S_list_Hat_T = []
        for i in range(stage_num):
            temp = S_list[i]
            self.S_list_Hat_T.append((temp/ (torch.sum(temp, 0, keepdim=True, dtype=torch.float32))).t())  # Q
        self.S_list.reverse() 
        for idx in range(len(self.S_list)):
            print(self.S_list[idx].shape)

        ################ PSFormer Architecture ################
        ## 1.stem module
        self.stem = nn.Sequential(
            nn.BatchNorm2d(channel), 
            nn.Conv2d(channel, layers_per_depth[0], 1), 
            nn.LeakyReLU())

        ## 2.transformer backbone
        #### a patch embed layer
        self.patch_embed1 = bn_1dconv_lrelu(128, 64)           
        self.patch_embed2 = bn_1dconv_lrelu(64, 32)         
        self.patch_embed3 = bn_1dconv_lrelu(32, 16)   
        self.patch_embed4 = bn_1dconv_lrelu(16, 8)
        self.patch_embed5 = bn_1dconv_lrelu(8, 4)

        #### Li transformer encoder layers
        self.block1=nn.Sequential()        
        for i in range(num_layer[0]):
            self.block1.add_module('block1_'+str(i), Encoder_layer(
                dim=64, num_head=4, mlp_ratio=1))
        
        self.block2=nn.Sequential()        
        for i in range(num_layer[1]):
            self.block2.add_module('block2_'+str(i), Encoder_layer(
                dim=32, num_head=4, mlp_ratio=1))
        
        self.block3=nn.Sequential()       
        for i in range(num_layer[2]):
            self.block3.add_module('block3_'+str(i), Encoder_layer(
                dim=16, num_head=4, mlp_ratio=1))
        
        self.block4=nn.Sequential()
        for i in range(num_layer[3]):
            self.block4.add_module('block4_'+str(i), Encoder_layer(
                dim=8, num_head=4, mlp_ratio=1))

        self.block5=nn.Sequential()
        for i in range(num_layer[4]):
            self.block5.add_module('block5_'+str(i), Encoder_layer(
                dim=4, num_head=4, mlp_ratio=1))

        ## 3.classification head
        # #### scheme 1: add
        # self.fc_add = nn.Linear(layers_per_depth[0], class_count)
        # #### scheme 2: concate
        # self.fc_cat = nn.Linear(layers_per_depth[0]*5, class_count)
        #### scheme 3: gate mechanism
        self.convblock = nn.Sequential()
        for i in range(stage_num+1):
            dim = layers_per_depth[-i-1]
            self.convblock.add_module('convblock_'+str(i), 
                bn_bsconv_lrelu(dim, layers_per_depth[0], kernel_size=3))
                # nn.Sequential(nn.BatchNorm2d(dim), nn.Conv2d(dim, layers_per_depth[0], 3, padding=1, bias=False), nn.LeakyReLU()))

        self.gate = Gate(layers_per_depth[0])
        self.fc = nn.Linear(layers_per_depth[0], class_count)

        self.Softmax = nn.Softmax(-1)

    def forward(self, x):
        (h, w, c) = x.shape
        x = torch.unsqueeze(x.permute([2, 0, 1]), 0)
        
        ## 1.stem
        x = self.stem(x)
        x = x[0].permute(1, 2, 0).reshape([h * w, -1])
        x0 = x ## [M0, C] or [N, C] 

        ## 2.transformer backbone
        if self.stage_num == 1:
            ## stage 1 ##
            x = torch.mm(self.S_list_Hat_T[0], x)
            x = self.patch_embed1(x)
            x = self.block1(x)
            x1 = x ## [M1, C]
            encoder_features = [x0, x1]

        if self.stage_num == 2:
            ## stage 1 ##
            x = torch.mm(self.S_list_Hat_T[0], x)
            x = self.patch_embed1(x)
            x = self.block1(x)
            x1 = x ## [M1, C]

            x = torch.mm(self.S_list_Hat_T[1], x)
            x = self.patch_embed2(x)
            x = self.block2(x)
            x2 = x ## [M2, C]
            encoder_features = [x0, x1, x2]

        if self.stage_num == 3:
            ## stage 1 ##
            x = torch.mm(self.S_list_Hat_T[0], x)
            x = self.patch_embed1(x)
            x = self.block1(x)
            x1 = x ## [M1, C]

            x = torch.mm(self.S_list_Hat_T[1], x)
            x = self.patch_embed2(x)
            x = self.block2(x)
            x2 = x ## [M2, C]

            x = torch.mm(self.S_list_Hat_T[2], x)
            x = self.patch_embed3(x)
            x = self.block3(x)
            x3 = x ## [M3, C]
            encoder_features = [x0, x1, x2, x3]

        if self.stage_num == 4:
            for i in range(1):
                x = torch.mm(self.S_list_Hat_T[i], x)
            ## stage 1 ##
            # x = torch.mm(self.S_list_Hat_T[0], x)
            x = self.patch_embed1(x)
            x = self.block1(x)
            x1 = x ## [M1, C]

            # x = torch.mm(self.S_list_Hat_T[1], x)
            x = self.patch_embed2(x)
            x = self.block2(x)
            x2 = x ## [M2, C]

            # x = torch.mm(self.S_list_Hat_T[2], x)
            x = self.patch_embed3(x)
            x = self.block3(x)
            x3 = x ## [M3, C]

            # x = torch.mm(self.S_list_Hat_T[3], x)
            x = self.patch_embed4(x)
            x = self.block4(x)
            x4 = x ## [M4, C]
            encoder_features = [x0, x1, x2, x3, x4]

        # if self.stage_num == 5:
        #     ## stage 1 ##
        #     x = torch.mm(self.S_list_Hat_T[0], x)
        #     x = self.patch_embed1(x)
        #     x = self.block1(x)
        #     x1 = x ## [M1, C]

        #     x = torch.mm(self.S_list_Hat_T[1], x)
        #     x = self.patch_embed2(x)
        #     x = self.block2(x)
        #     x2 = x ## [M2, C]

        #     x = torch.mm(self.S_list_Hat_T[2], x)
        #     x = self.patch_embed3(x)
        #     x = self.block3(x)
        #     x3 = x ## [M3, C]

        #     x = torch.mm(self.S_list_Hat_T[3], x)
        #     x = self.patch_embed4(x)
        #     x = self.block4(x)
        #     x4 = x ## [M4, C]

        #     x = torch.mm(self.S_list_Hat_T[4], x)
        #     x = self.patch_embed5(x)
        #     x = self.block5(x)
        #     x5 = x ## [M5, C]
        #     encoder_features = [x0, x1, x2, x3, x4, x5]

        ### 3.classification head
        decoder_features = []
        encoder_features.reverse() 
        for i in range(len(encoder_features)):
            tmp = encoder_features[i]
            # for j in range(len(encoder_features)-1-i):
            #     tmp = torch.mm(self.S_list[j+i], tmp)
            if i == len(encoder_features)-1:
                pass
            else:
                for j in range(1):
                    # print(self.S_list[j].shape, tmp.shape)
                    tmp = torch.mm(self.S_list[j-1], tmp)
            tmp = tmp.reshape([1,h,w,-1]).permute([0,3,1,2]) 
            tmp = self.convblock[i](tmp)[0].permute([1, 2, 0]).reshape([h * w, -1]) #[H*W, C] ## 3*3 BSConv
            decoder_features.append(tmp)
        decoder_features.reverse()  #[x0, x1, x2, x3, x4]

        #### gate mechanism
        x = torch.stack(decoder_features, dim=1) #[H*W, C]
        x, attn = self.gate(x) ## gate mechanism
        x = self.Softmax(self.fc(x)) ## linear + softmax

        return x, 0
