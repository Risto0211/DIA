
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
# from ddpm_conditional import obtain_same_set, Diffusion, init_parser
from utils import load_checkpoint
from net import CIFAR10CNN
from inspect import isfunction
from einops import rearrange, repeat
from torch import einsum


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        #print (f"in sa: {x.shape}")
        #print (self.channels, self.size, self.size * self.size)

        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
    

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.ff_self = nn.Sequential(
            nn.LayerNorm([query_dim]),
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )

    def forward(self, x, context=None, mask=None):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        cb, cc, ch, cw = context.shape
        context = rearrange(context, 'cb cc ch cw -> cb (ch cw) cc')
        head = self.heads

        q = self.to_q(x)
        context = default(context, x)
        # print(context.shape)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=head), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=head)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=head)
        out = self.to_out(out) # attn
        out += x # res
        out = self.ff_self(out) + out
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        return out



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class accomp_Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, accompany_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x, accompany_x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    

class cond_Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.up(x)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 16)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 8)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 4)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 8)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 16)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 32)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)

        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, num_classes=None, device="cuda", target_dim=None, args=None,):
        super().__init__()
        self.args=args
        self.adjust = 0
        if target_dim:
            B, C, H, W=target_dim
            self.target_C=C
            self.target_H=H
            self.target_W=W
        #print (self.target_C, self.target_H, self.target_W)


        if args.condition_method == 'add_channel':
            c_in+=3

        self.device = device
        self.time_dim = args.time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, emb_dim=self.time_dim)
        self.sa1 = SelfAttention(128, self.args.image_size//2)
        self.down2 = Down(128, 256, emb_dim=self.time_dim)
        self.sa2 = SelfAttention(256, self.args.image_size//4)
        self.down3 = Down(256, 256, emb_dim=self.time_dim)
        self.sa3 = SelfAttention(256, self.args.image_size//8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, emb_dim=self.time_dim)
        self.sa4 = SelfAttention(128, self.args.image_size//4)
        self.up2 = Up(256, 64, emb_dim=self.time_dim)
        self.sa5 = SelfAttention(64, self.args.image_size//2)
        self.up3 = Up(128, 64, emb_dim=self.time_dim)
        self.sa6 = SelfAttention(64, self.args.image_size)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        #if num_classes is not None:
        if args.class_condition:
            self.class_condition_encoder = nn.Embedding(num_classes, args.time_dim)

        if args.target_model == 'vit':
            self.adjust = 1
            n = self.target_H*self.target_W
            self.adjust_linear = nn.Linear(1+n, n)

        if args.condition=='intermediate_feature_map':
            if args.encoder == 'small_inverse_model':
                self.condition_encoder = small_inverse_model(num_classes, args.time_dim, self.target_C, self.target_H, self.target_W, args)
            elif args.encoder == 'linear':
                self.condition_encoder = mapping_mlp(args, args.time_dim, self.target_C, self.target_H, self.target_W)

                #if args.condition_method=='add_t_emb':
                #    self.condition_encoder = mapping_mlp(self.args, time_dim)
                #elif args.condition_method=='add_channel':
                #    self.condition_encoder = small_inverse_model(num_classes, args)
                #    print ("condition_encoder", self.condition_encoder)


            #elif args.condition=='logits':
            #    if args.condition_method=='add_t_emb':
            #        self.condition_encoder = mapping_mlp(self.args, time_dim)
            #    elif args.condition_method=='add_channel':
            #        self.condition_encoder = small_inverse_model(num_classes, args)
            #        print ("condition_encoder", self.condition_encoder)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y, intermediate_feature_map=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(t.shape)
        # print(self.label_emb(y).shape)
        if self.args.class_condition and y is not None:
            t += self.args.class_condition_coef*self.class_condition_encoder(y)

        if self.adjust == 1:
            b, n, c = intermediate_feature_map.shape
            intermediate_feature_map = self.adjust_linear(intermediate_feature_map.permute(0, 2, 1).reshape(b*c, n))
            intermediate_feature_map = intermediate_feature_map.reshape(b, c, int((n-1)**0.5), int((n-1)**0.5))


        if self.args.condition_method == 'add_t_emb' and intermediate_feature_map is not None:
            t += self.args.feature_map_condition_coef*self.condition_encoder(intermediate_feature_map)

        elif self.args.condition_method=='add_channel':
            if intermediate_feature_map is not None:
                cur_condition=self.condition_encoder(intermediate_feature_map)
                x=torch.cat([cur_condition, x], dim=1) # B, C, H, W

        #print (f"x.shape:{x.shape}")
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        #print (f"x2.shape:{x2.shape}")
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    

class UNet_conditional_CrossAttention(nn.Module):
    def __init__(self, c_in=3, c_out=3, num_classes=None, device="cuda", target_dim=None, args=None,):
        super().__init__()
        self.args=args
        self.adjust = 0
        self.c_in = c_in
        if target_dim:
            B, C, H, W=target_dim
            self.target_C=C
            self.target_H=H
            self.target_W=W
        #print (self.target_C, self.target_H, self.target_W)

        if args.condition=='intermediate_feature_map':
            if args.encoder == 'small_inverse_model':
                self.condition_encoder = small_inverse_model(num_classes, args.time_dim, self.target_C, self.target_H, self.target_W, args)

        self.target_C = args.channel

        self.device = device
        self.time_dim = args.time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, emb_dim=self.time_dim)
        self.ca1 = CrossAttention(query_dim=128, context_dim=self.target_C)
        self.down2 = Down(128, 256, emb_dim=self.time_dim)
        self.ca2 = CrossAttention(query_dim=256, context_dim=self.target_C)
        self.down3 = Down(256, 256, emb_dim=self.time_dim)
        self.ca3 = CrossAttention(query_dim=256, context_dim=self.target_C)
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, emb_dim=self.time_dim)
        self.ca4 = CrossAttention(query_dim=128, context_dim=self.target_C)
        self.up2 = Up(256, 64, emb_dim=self.time_dim)
        self.ca5 = CrossAttention(query_dim=64, context_dim=self.target_C)
        self.up3 = Up(128, 64, emb_dim=self.time_dim)
        self.ca6 = CrossAttention(query_dim=64, context_dim=self.target_C)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        #if num_classes is not None:
        if args.class_condition:
            self.class_condition_encoder = nn.Embedding(num_classes, args.time_dim)

        if args.target_model == 'vit':
            self.adjust = 1
            n = self.target_H*self.target_W
            self.adjust_linear = nn.Linear(1+n, n)


        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y, intermediate_feature_map=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(t.shape)
        # print(self.label_emb(y).shape)
        if self.args.class_condition and y is not None:
            t += self.args.class_condition_coef*self.class_condition_encoder(y)

        if self.adjust == 1:
            b, n, c = intermediate_feature_map.shape
            intermediate_feature_map = self.adjust_linear(intermediate_feature_map.permute(0, 2, 1).reshape(b*c, n))
            intermediate_feature_map = intermediate_feature_map.reshape(b, c, int((n-1)**0.5), int((n-1)**0.5))

        intermediate_feature_map = self.condition_encoder(intermediate_feature_map)

        #print (f"x.shape:{x.shape}")
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        #print (f"x2.shape:{x2.shape}")
        x2 = self.ca1(x2, intermediate_feature_map)
        x3 = self.down2(x2, t)
        x3 = self.ca2(x3, intermediate_feature_map)
        x4 = self.down3(x3, t)
        x4 = self.ca3(x4, intermediate_feature_map)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.ca4(x, intermediate_feature_map)
        x = self.up2(x, x2, t)
        x = self.ca5(x, intermediate_feature_map)
        x = self.up3(x, x1, t)
        x = self.ca6(x, intermediate_feature_map)
        output = self.outc(x)
        return output



class Double_UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, num_classes=None, device="cuda", target_dim=None, args=None,):
        super().__init__()
        self.args=args
        self.c_in = c_in
        self.adjust = 0
        if target_dim:
            B, C, H, W=target_dim
            self.target_C=C
            self.target_H=H
            self.target_W=W
        #print (self.target_C, self.target_H, self.target_W)


        # if args.condition_method == 'add_channel':
        #     c_in+=3

        self.device = device
        self.time_dim = args.time_dim
        self.inc = DoubleConv(c_in, 64)
        self.cond_inc = DoubleConv(c_in, 32)
        self.down1 = Down(96, 128, emb_dim=self.time_dim)
        self.cond_down1 = Down(32, 64, emb_dim=self.time_dim)
        self.sa1 = SelfAttention(128, self.args.image_size//2)
        self.down2 = Down(192, 256, emb_dim=self.time_dim)
        self.cond_down2 = Down(64, 128, emb_dim=self.time_dim)
        self.sa2 = SelfAttention(256, self.args.image_size//4)
        self.down3 = Down(384, 512, emb_dim=self.time_dim)
        self.cond_down3 = Down(128, 256, emb_dim=self.time_dim)
        self.sa3 = SelfAttention(512, self.args.image_size//8)

        self.bot1 = DoubleConv(768, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = accomp_Up(768, 128, emb_dim=self.time_dim)
        self.cond_up1 = cond_Up(256, 128, emb_dim=self.time_dim)
        self.sa4 = SelfAttention(128, self.args.image_size//4)
        self.up2 = accomp_Up(384, 64, emb_dim=self.time_dim)
        self.cond_up2 = cond_Up(128, 64, emb_dim=self.time_dim)
        self.sa5 = SelfAttention(64, self.args.image_size//2)
        self.up3 = accomp_Up(192, 64, emb_dim=self.time_dim)
        self.cond_up3 = cond_Up(64, 32, emb_dim=self.time_dim)
        self.sa6 = SelfAttention(64, self.args.image_size)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        #if num_classes is not None:
        if args.class_condition:
            self.class_condition_encoder = nn.Embedding(num_classes, args.time_dim)

        if args.target_model == 'vit':
            self.adjust = 1
            n = self.target_H*self.target_W
            self.adjust_linear = nn.Linear(1+n, n)

        if args.condition=='intermediate_feature_map':
            if args.encoder == 'small_inverse_model':
                self.condition_encoder = small_inverse_model(num_classes, args.time_dim, self.target_C, self.target_H, self.target_W, args)
            elif args.encoder == 'linear':
                self.condition_encoder = mapping_mlp(args, args.time_dim, self.target_C, self.target_H, self.target_W)

        

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y, intermediate_feature_map=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(t.shape)
        # print(self.label_emb(y).shape)
        if self.args.class_condition and y is not None:
            t += self.args.class_condition_coef*self.class_condition_encoder(y)

        if self.adjust == 1:
            b, n, c = intermediate_feature_map.shape
            intermediate_feature_map = self.adjust_linear(intermediate_feature_map.permute(0, 2, 1).reshape(b*c, n))
            intermediate_feature_map = intermediate_feature_map.reshape(b, c, int((n-1)**0.5), int((n-1)**0.5))

        feat_map = self.condition_encoder(intermediate_feature_map)
        # print(feat_map.shape)

        #print (f"x.shape:{x.shape}")
        x1 = self.inc(x) # B C H W
        feat_map1 = self.cond_inc(feat_map) # B C H W
        x1 = torch.cat([x1, feat_map1], dim=1) # B 2*C H W
        # print(x1.shape)
        x2 = self.down1(x1, t)
        feat_map2 = self.cond_down1(feat_map1, t)
        #print (f"x2.shape:{x2.shape}")
        x2 = self.sa1(x2)
        x2 = torch.cat([x2, feat_map2], dim=1)
        # print(x2.shape)
        x3 = self.down2(x2, t)
        feat_map3 = self.cond_down2(feat_map2, t)
        x3 = self.sa2(x3)
        x3 = torch.cat([x3, feat_map3], dim=1)
        # print(x3.shape)
        x4 = self.down3(x3, t)
        feat_map4 = self.cond_down3(feat_map3, t)
        x4 = self.sa3(x4)
        x4 = torch.cat([x4, feat_map4], dim=1)
        # print(x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # x = self.up1(x4, x3, t)
        feat_map = self.cond_up1(feat_map4, t)
        x = self.up1(x4, x3, feat_map, t)
        # x = self.up1(x4, x3, feat_map3, t)
        x = self.sa4(x)
        feat_map = self.cond_up2(feat_map, t)
        x = self.up2(x, x2, feat_map, t)
        # x = self.up2(x, x2, feat_map2, t)
        x = self.sa5(x)
        feat_map = self.cond_up3(feat_map, t)
        x = self.up3(x, x1, feat_map, t)
        # x = self.up3(x, x1, feat_map1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class Double_UNet_conditional_tiny(nn.Module):
    def __init__(self, c_in=3, c_out=3, num_classes=None, device="cuda", target_dim=None, args=None,):
        super().__init__()
        self.args=args
        if target_dim:
            B, C, H, W=target_dim
            self.target_C=C
            self.target_H=H
            self.target_W=W
        #print (self.target_C, self.target_H, self.target_W)


        # if args.condition_method == 'add_channel':
        #     c_in+=3

        self.device = device
        self.time_dim = args.time_dim
        self.inc = DoubleConv(c_in, 64//2)
        self.cond_inc = DoubleConv(c_in, 32//2)
        self.down1 = Down(96//2, 128//2, emb_dim=self.time_dim)
        self.cond_down1 = Down(32//2, 64//2, emb_dim=self.time_dim)
        self.sa1 = SelfAttention(128//2, self.args.image_size//2)
        self.down2 = Down(192//2, 256//2, emb_dim=self.time_dim)
        self.cond_down2 = Down(64//2, 128//2, emb_dim=self.time_dim)
        self.sa2 = SelfAttention(256//2, self.args.image_size//4)
        self.down3 = Down(384//2, 512//2, emb_dim=self.time_dim)
        self.cond_down3 = Down(128//2, 256//2, emb_dim=self.time_dim)
        self.sa3 = SelfAttention(512//2, self.args.image_size//8)

        self.bot1 = DoubleConv(768//2, 512//2)
        self.bot2 = DoubleConv(512//2, 512//2)
        self.bot3 = DoubleConv(512//2, 256//2)

        self.up1 = accomp_Up(768//2, 128//2, emb_dim=self.time_dim)
        self.cond_up1 = cond_Up(256//2, 128//2, emb_dim=self.time_dim)
        self.sa4 = SelfAttention(128//2, self.args.image_size//4)
        self.up2 = accomp_Up(384//2, 64//2, emb_dim=self.time_dim)
        self.cond_up2 = cond_Up(128//2, 64//2, emb_dim=self.time_dim)
        self.sa5 = SelfAttention(64//2, self.args.image_size//2)
        self.up3 = accomp_Up(192//2, 64//2, emb_dim=self.time_dim)
        self.cond_up3 = cond_Up(64//2, 32//2, emb_dim=self.time_dim)
        self.sa6 = SelfAttention(64//2, self.args.image_size)
        self.outc = nn.Conv2d(64//2, c_out, kernel_size=1)

        #if num_classes is not None:
        if args.class_condition:
            self.class_condition_encoder = nn.Embedding(num_classes, args.time_dim)

        if args.condition=='intermediate_feature_map':
            if args.encoder == 'small_inverse_model':
                self.condition_encoder = small_inverse_model(num_classes, args.time_dim, self.target_C, self.target_H, self.target_W, args)
            elif args.encoder == 'linear':
                self.condition_encoder = mapping_mlp(args, args.time_dim, self.target_C, self.target_H, self.target_W)

                #if args.condition_method=='add_t_emb':
                #    self.condition_encoder = mapping_mlp(self.args, time_dim)
                #elif args.condition_method=='add_channel':
                #    self.condition_encoder = small_inverse_model(num_classes, args)
                #    print ("condition_encoder", self.condition_encoder)


            #elif args.condition=='logits':
            #    if args.condition_method=='add_t_emb':
            #        self.condition_encoder = mapping_mlp(self.args, time_dim)
            #    elif args.condition_method=='add_channel':
            #        self.condition_encoder = small_inverse_model(num_classes, args)
            #        print ("condition_encoder", self.condition_encoder)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y, intermediate_feature_map=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(t.shape)
        # print(self.label_emb(y).shape)
        if self.args.class_condition and y is not None:
            t += self.args.class_condition_coef*self.class_condition_encoder(y)

        feat_map = self.condition_encoder(intermediate_feature_map)
        # print(feat_map.shape)

        #print (f"x.shape:{x.shape}")
        x1 = self.inc(x) # B C H W
        feat_map = self.cond_inc(feat_map) # B C H W
        x1 = torch.cat([x1, feat_map], dim=1) # B 2*C H W
        # print(x1.shape)
        x2 = self.down1(x1, t)
        feat_map = self.cond_down1(feat_map, t)
        #print (f"x2.shape:{x2.shape}")
        x2 = self.sa1(x2)
        x2 = torch.cat([x2, feat_map], dim=1)
        # print(x2.shape)
        x3 = self.down2(x2, t)
        feat_map = self.cond_down2(feat_map, t)
        x3 = self.sa2(x3)
        x3 = torch.cat([x3, feat_map], dim=1)
        # print(x3.shape)
        x4 = self.down3(x3, t)
        feat_map = self.cond_down3(feat_map, t)
        x4 = self.sa3(x4)
        x4 = torch.cat([x4, feat_map], dim=1)
        # print(x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # x = self.up1(x4, x3, t)
        feat_map = self.cond_up1(feat_map, t)
        x = self.up1(x4, x3, feat_map, t)
        x = self.sa4(x)
        feat_map = self.cond_up2(feat_map, t)
        x = self.up2(x, x2, feat_map, t)
        x = self.sa5(x)
        feat_map = self.cond_up3(feat_map, t)
        x = self.up3(x, x1, feat_map, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class mapping_mlp(nn.Module):
    def __init__(self, args, num_dim, target_C=None, target_H=None, target_W=None):
        super().__init__()
        num_classes = target_C*target_H*target_W
        self.method = args.add_t_emb_structure
        if self.method == 'linear':
            self.fc1 = nn.Linear(num_classes, 4*num_dim)
            self.fc2 = nn.Linear(4*num_dim, 2*num_dim)
            self.fc3 = nn.Linear(2*num_dim, num_dim)

        elif self.method == 'tcnn':
            # B*length*1*1
            self.tconv11 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=32, kernel_size=4, stride=1, padding=0)
            # B*32*4*4
            self.tconv12 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=6, stride=4, padding=1)
            # B*16*16*16
            self.conv11 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
            # B*1*16*16
            self.conv_fc1 = nn.Linear(16*16, num_dim)

        elif self.method == 'attn':
            self.nhead = 8
            self.tconv21 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=32, kernel_size=4, stride=1, padding=0)
            # B*32*4*4
            self.tconv22 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=6, stride=4, padding=1)
            # B*16*16*16
            self.conv21 = nn.Conv2d(in_channels=16, out_channels=4*4, kernel_size=4, stride=4, padding=0)
            # B*16*4*4

            self.conv_fc2 = nn.Linear(16*16, num_dim)
            
            self.mha = nn.MultiheadAttention(embed_dim=4*4, num_heads=self.nhead)


    def forward(self, x):
        if self.method == 'linear':
            x = torch.flatten(x, 1)
            out = self.fc1(x)
            out = self.fc2(out)
            out = self.fc3(out)

        elif self.method == 'tcnn':
            out = x.reshape(x.shape[0], x.shape[1], 1, 1)
            # print(f'out_shape={out.shape}')
            out = self.tconv11(out)
            # print(f'out_shape={out.shape}')
            out = self.tconv12(out)
            # print(f'out_shape={out.shape}')
            out = self.conv11(out)
            # print(f'out_shape={out.shape}')
            out = out.reshape(out.shape[0], -1)
            # print(f'out_shape={out.shape}')
            out = self.conv_fc1(out)
            # print(f'out_shape={out.shape}')

        elif self.method == 'attn':
            out = x.reshape(x.shape[0], x.shape[1], 1, 1)
            # print(f'out_shape={out.shape}')
            out = self.tconv21(out)
            # print(f'out_shape={out.shape}')
            out = self.tconv22(out)
            # print(f'out_shape={out.shape}')
            out = self.conv21(out)
            # print(f'out_shape={out.shape}')
            out = out.reshape(out.shape[0], out.shape[1], -1).permute(1, 0, 2)
            # print(f'out_shape={out.shape}')
            out, weights = self.mha(out, out, out)
            # print(f'out_shape={out.shape}')
            out = out.transpose(0, 1)
            out = out.reshape(out.shape[0], -1)
            # print(f'out_shape={out.shape}')
            out = self.conv_fc2(out)
            # print(f'out_shape={out.shape}')

        return out
    

class small_inverse_model(nn.Module):
    def __init__(self, num_classes, time_dim=None,target_C=None, target_H=None, target_W=None, args=None, stride=1):
        super().__init__()
        self.args=args

        # first double mutiple times to get the same size as image
        # self.num_double_layer=(args.image_size//target_W)-1
        self.num_double_layer=(torch.log2(torch.tensor(args.image_size//target_W))).int()
        print (f"num_double_layer:{self.num_double_layer}")

        # same size, for better performance
        self.num_enhance_layer=args.condition_encoder_size
        print (f"num_enhance_layer:{self.num_enhance_layer}")

        cur_C=target_C
        self.double_layers = nn.ModuleList()
        for _ in range(self.num_double_layer):
            #self.dilate_layers[lidx] = self._make_layers(3, 3, "double")
            self.double_layers.append(self._make_layers(cur_C, cur_C//2, "double"))
            cur_C//=2

        self.enhance_layers = nn.ModuleList()
        for _ in range(self.num_enhance_layer):

            self.enhance_layers.append(self._make_layers(cur_C, cur_C, "same"))
            self.enhance_layers.append(BasicBlock(cur_C, cur_C, stride))

        self.enhance_layers.append(self._make_layers(cur_C, args.channel, "same")) # shrink the condition

        if args.condition_method == 'add_t_emb':
            self.cur_C=args.channel
            self.fc1 = nn.Linear(self.cur_C*args.image_size*args.image_size, time_dim)
            #self.layers.append(self.fc1)


    @staticmethod
    def _make_layers(InChannels, OutChannels, csizechange):
        # generic 1 layer ConvTranspose2d follow: (CIFAR10 Input image is 32*32)32=(Hin-1)*S+K
        S=  {'double': 2, 'same': 1}
        K = {'double': 3, 'same': 3}
        OP ={'double': 1, 'same': 0}
        #print (InChannels, OutChannels, csizechange)
        deconv11 = nn.ConvTranspose2d(
            in_channels = InChannels,
            out_channels = OutChannels,
            #kernel_size = K[intermediate_size],
            kernel_size=K[csizechange],
            padding = 1,
            output_padding = OP[csizechange],
            stride=S[csizechange]
        )
        #self.layerDict['deconv11'] = self.deconv11
        return deconv11

    def forward(self, x):
        B, C, H, W=x.shape
        # print (f"intermediate_feature_map x: {x.shape}")

        for lidx, layer in enumerate(self.double_layers):
            x = layer(x)
            # print (f"intermediate_feature_map x: {x.shape}")

        assert x.shape[2] == x.shape[3] == self.args.image_size


        for lidx, layer in enumerate(self.enhance_layers):
            x = layer(x)
            #print (f"intermediate_feature_map x in enhance layer: {x.shape}")

        if self.args.condition_method == 'add_t_emb':
            x = x.reshape(B, self.cur_C*self.args.image_size*self.args.image_size)
            #print (f"intermediate_feature_map x: {x.shape}")
            x = self.fc1(x)
            #print (f"intermediate_feature_map x: {x.shape}")




        return x


class small_inverse_model_logits(nn.Module):
    def __init__(self, num_classes, args=None, stride=1):
        super().__init__()
        self.args=args
        #self.output_shape=(3, args.image_size, args.image_size)
        #self.fc = nn.Linear(num_classes, self.output_shape[0]*self.output_shape[1]*self.output_shape[2])
        # linearly map to a smaller version first
        self.condition_encoder_size=args.condition_encoder_size
        self.output_shape = (args.channel, args.image_size//(2**self.condition_encoder_size), args.image_size//(2**self.condition_encoder_size))
        print (f"output_shape:{self.output_shape}")
        self.fc1= nn.Linear(num_classes, self.output_shape[0]*self.output_shape[1]*self.output_shape[2])
        self.dilate_layers = nn.ModuleList()
        #self.dilate_layers=[None for _ in range(self.condition_encoder_size)]
        for lidx in range(self.condition_encoder_size):
            #self.dilate_layers[lidx] = self._make_layers(3, 3, "double")
            self.dilate_layers.append(self._make_layers(3, 3, "double"))
        self.dilate_layers.append(BasicBlock(3, 3, stride))


    @staticmethod
    def _make_layers(InChannels, OutChannels, csizechange):
        # generic 1 layer ConvTranspose2d follow: (CIFAR10 Input image is 32*32)32=(Hin-1)*S+K
        S=  {'double': 2, 'same': 1}
        K = {'double': 3, 'same': 3}
        OP ={'double': 1, 'same': 0}
        #print (InChannels, OutChannels, csizechange)
        deconv11 = nn.ConvTranspose2d(
            in_channels = InChannels,
            out_channels = OutChannels,
            #kernel_size = K[intermediate_size],
            kernel_size=K[csizechange],
            padding = 1,
            output_padding = OP[csizechange],
            stride=S[csizechange]
        )
        #self.layerDict['deconv11'] = self.deconv11
        return deconv11
    def forward(self, x):
        B, num_class=x.shape
        #print (f"input x: {x.shape}")
        x = self.fc1(x)
        #print (f" x: {x.shape}")
        x = x.reshape(B, self.output_shape[0], self.output_shape[1], self.output_shape[2])
        #print (f"x: {x.shape}")
        for lidx, layer in enumerate(self.dilate_layers):
            x = layer(x)
            #print (f"x: {x.shape}")
        assert x.shape[2]==x.shape[3]==self.args.image_size
        return x
    

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        #shortcut
        self.shortcut = nn.Sequential()
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    def get_residual_output(self, x, target_layer):
        for layer in self.residual_function:
            x = layer(x)
            if layer == target_layer:
                return x
        print("Target layer not found")
        exit(1)
    def get_shortcut_output(self, x, target_layer):
        for layer in self.shortcut:
            x = layer(x)
            if layer == target_layer:
                return x
        print("Target layer not found")
        exit(1)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='DDPM_conditional')
    # parser = init_parser(parser)
    # args = parser.parse_args()
    # device = args.device
    # args.num_classes = 10
    # trainset, testset, inv_normalize = obtain_same_set(args)
    # trainloader = torch.utils.data.DataLoader(
    #         trainset, batch_size=args.batch_size, shuffle=True,num_workers=2)
    # print ("load target model")

    # target_model=CIFAR10CNN(3)
    # target_model_state_dict, target_optimizer_state_dict = load_checkpoint(args.target_model_path)
    # target_model.load_state_dict(target_model_state_dict)
    # target_model = target_model.to(device)

    # train_iter_single = iter(trainloader)
    # single_batch = next(train_iter_single)
    # single_inputs, single_labels = single_batch
    # single_inputs=single_inputs.to(device)

    # target_dim, target_depth=target_model.getLayerDepthandDim(single_inputs, target_model.layerDict[args.target_layer])
    # net = Double_UNet_conditional(num_classes=args.num_classes, target_dim=target_dim, args=args).to(device)
   
    # x = single_inputs

    # target_model_feature_map = target_model.getLayerOutput(x, target_model.layerDict[args.target_layer]).clone()
    # # print(target_model_feature_map.shape)
    # diffusion = Diffusion(img_size=args.image_size, device=device)
    # t = diffusion.sample_timesteps(x.shape[0]).to(device)
    # x_t, noise = diffusion.noise_images(x, t)
    # y = None
    # predicted_noise = net(x_t, t, y, target_model_feature_map)

    pass