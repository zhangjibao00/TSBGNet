import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .external_function import SpectralNorm
from util import task


######################################################################################
# base function for network structure
######################################################################################


def init_weights(net, init_type='normal', gain=0.02):
    """Get different initial method for the network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            init.orthogonal_(m.weight.data, gain=gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.iter_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_net(net, init_type='normal', gpu_ids=[]):
    """print the network structure and initial the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f M' % (num_params / 1e6))

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True



######################################################################################
# Network basic function
######################################################################################
class Cooord_Attn(nn.Module):


    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(Cooord_Attn, self).__init__()
        self.input_nc = input_nc
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        self.softmax = nn.Softmax(dim=-1)
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.x_q_conv = SpectralNorm(nn.Conv2d(input_nc, input_nc, kernel_size=(1,1)))
        self.x_k_conv = SpectralNorm(nn.Conv2d(input_nc, input_nc, kernel_size=(1,1)))
        self.x_v_conv = SpectralNorm(nn.Conv2d(input_nc, input_nc, kernel_size=(1,1)))
        self.conv1 = SpectralNorm(nn.Conv2d(1, 1, kernel_size=(1,1)))

        self.guide_q_conv = SpectralNorm(nn.Conv2d(input_nc, input_nc, kernel_size=(1,1)))
        self.guide_k_conv = SpectralNorm(nn.Conv2d(input_nc, input_nc, kernel_size=(1,1)))
        self.guide_v_conv = SpectralNorm(nn.Conv2d(input_nc, input_nc, kernel_size=(1,1)))
        self.conv2 = SpectralNorm(nn.Conv2d(1, 1, kernel_size=(1,1)))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.conv1_attn = SpectralNorm(nn.Conv2d(input_nc, input_nc, **kwargs))
        self.conv2_attn = SpectralNorm(nn.Conv2d(input_nc, input_nc, **kwargs))
        self.shortcut = SpectralNorm(nn.Conv2d(input_nc, input_nc, kernel_size=(1, 1), stride=(1, 1)))
        self.model = nn.Sequential(self.nonlinearity, self.conv1_attn, self.nonlinearity, self.conv2_attn)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.channel_attn = nn.Sequential(self.pool, self.linear, self.nonlinearity, self.linear, self.sigmoid)
        self.coord = CoordConv(input_nc, input_nc, with_r=False)

    def forward(self, x, guide):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        B, C, W, H = x.size()
        x_channel_attn = self.channel_attn(x)
        x_coord = self.coord(x)
        x = torch.mul(x_channel_attn, x_coord)
        x_query = self.x_q_conv(x).view(B, -1, W * H).permute(0, 2, 1)  # B X (N)X C
        x_key = self.x_k_conv(x).view(B, -1, W * H)  # B X C x (N)
        x_energy = torch.bmm(x_query, x_key)  # transpose check
        x_attention = self.softmax(x_energy)  # BX (N) X (N)
        x_value = self.x_v_conv(x).view(B, -1, W * H)  # B X C X N
        x_out = torch.bmm(x_value, x_attention.permute(0, 2, 1))
        x_out = x_out.view(B, C, W, H)

        guide_channel_attn = self.channel_attn(guide)
        guide_coord = self.coord(guide)
        guide = torch.mul(guide_channel_attn, guide_coord)
        guide_query = self.guide_q_conv(guide).view(B, -1, W * H).permute(0, 2, 1)  # B X (N)X C
        guide_key = self.guide_k_conv(guide).view(B, -1, W * H)  # B X C x (N)
        guide_energy = torch.bmm(guide_query, guide_key)  # transpose check
        guide_attention = self.softmax(guide_energy)  # BX (N) X (N)
        guide_out = torch.bmm(x_value, guide_attention.permute(0, 2, 1))
        guide_out = guide_out.view(B, C, W, H)

        out = self.gamma * x_out + self.alpha * guide_out
        out = self.model(out) + self.shortcut(out) * guide_out


        return out


def style_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
    return loss_value


def perceptual_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value

class CoordConv(nn.Module):
    """
    CoordConv operation
    """
    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = SpectralNorm(nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret

class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret



from torch import nn
import torch
from einops import rearrange
import functools
import math
import copy
import torch.nn.functional as F

class PositionEmbeddingLearned(nn.Module):
    """
    This is a learned version of the position embedding
    """
    def __init__(self, path_size,num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(path_size, num_pos_feats)
        self.col_embed = nn.Embedding(path_size, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x,):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i).unsqueeze(0).repeat(h, 1, 1)
        y_emb = self.row_embed(j).unsqueeze(1).repeat(1, w, 1)
        pos = (x_emb + y_emb).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

class PixelwiseNorm(nn.Module):
    def __init__(self, input_nc):
        super(PixelwiseNorm, self).__init__()
        self.init = False
        self.alpha = nn.Parameter(torch.ones(1, input_nc, 1, 1))

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        # x = x - x.mean(dim=1, keepdim=True)
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).rsqrt()  # [N1HW]
        y = x * y  # normalize the input x volume
        return self.alpha*y

class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information from different position"""
    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.bias = bias
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_out = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        if self.bias:
            nn.init.constant_(self.to_q.bias, 0.)
            nn.init.constant_(self.to_k.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self, q, k, v):
        h =self.num_heads
        # calculate similarity map
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q = rearrange(q, 'b n (h d)->b h n d', h=h)
        k = rearrange(k, 'b n (h d)->b h n d', h=h)
        v = rearrange(v, 'b n (h d)->b h n d', h=h)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # calculate the attention value
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        # projection
        out = torch.einsum('bhij, bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class juzheng(nn.Module):
    def __init__(self):
        super(juzheng, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_a = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        kwargs_b = {'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1}
        kwargs_c = {'kernel_size': 7, 'stride': 2, 'padding': 3, 'output_padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        kwargs_gate = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_out = {'kernel_size': 3, 'padding': 0, 'bias': True}
        # self.nonlinearity = nn.LeakyReLU(0.1)
        self.nonlinearity = nn.Tanh()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
        self.pool_fix = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.gate_nonlinearity = nn.Sigmoid()
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)

        # gate1
        self.conv18 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.conv19 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.model12 = nn.Sequential(self.conv19, self.gate_nonlinearity)
        self.model13 = nn.Sequential(self.conv18, self.gate_nonlinearity)


        # gate2
        self.convr = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.convr1 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.model4 = nn.Sequential(self.convr, self.nonlinearity)
        self.model5 = nn.Sequential(self.convr1, self.nonlinearity)



    def forward(self, st, te, ht_1):
        q1 = rearrange(st, 'b c h w->b c (h w)')
        q2 = rearrange(te, 'b c h w->b (h w) c')
        q = torch.bmm(q1, q2)

        hs = self.model4(torch.cat(((self.model12(torch.cat((st, ht_1), 1)) * ht_1), st), 1))
        ht = self.model5(torch.cat(((self.model13(torch.cat((te, ht_1), 1)) * ht_1), te), 1))

        hs = rearrange(hs, 'b c h w->b (h w) c')
        ht = rearrange(ht, 'b c h w->b (h w) c')

        h = torch.bmm(hs, q.transpose(1, 2)) + torch.bmm(ht, q)
        h = rearrange(h, 'b (h w) c->b c h w', h=32, w=32)

        return h
class Transformer(nn.Module):
    def __init__(self, embed_dim=128, output_nc=128, dim_conv=512, kernel=3, num_heads=4, dropout=0.,
                 path_size=32):
        super(Transformer, self).__init__()
        norm_layer = functools.partial(PixelwiseNorm)
        activation_layer = nn.GELU()
        self.pos_embeds = PositionEmbeddingLearned(path_size, embed_dim)
        self.pos_embedt = PositionEmbeddingLearned(path_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.encoder_trans = TransformerEncoderLayer(embed_dim, num_heads, dim_conv, kernel, dropout)
        self.encoder_trans_1 = TransformerEncoderLayer(embed_dim, num_heads, dim_conv, kernel, dropout)
        self.encoder_trans_2 = TransformerEncoderLayer(embed_dim, num_heads, dim_conv, kernel, dropout)
        self.encoder_trant = TransformerEncoderLayer(embed_dim, num_heads, dim_conv, kernel, dropout)
        self.encoder_trant_1 = TransformerEncoderLayer(embed_dim, num_heads, dim_conv, kernel, dropout)
        self.encoder_trant_2 = TransformerEncoderLayer(embed_dim, num_heads, dim_conv, kernel, dropout)

        self.juzheng = juzheng()

        self.to_token = nn.Sequential(
            norm_layer(embed_dim),
            activation_layer,
            nn.Conv2d(embed_dim, output_nc, kernel_size=1, stride=1, padding=0)
        )
        self.to_tokent = nn.Sequential(
            norm_layer(embed_dim),
            activation_layer,
            nn.Conv2d(embed_dim, output_nc, kernel_size=1, stride=1, padding=0)
        )
    # x:q 纹理  x1:k,v  结构
    def forward(self, x, x1, ht):
        # 纹理
        x_pos = self.pos_embeds(x)
        x_pos = rearrange(x_pos, 'b c h w -> b (h w) c')
        # 结构
        x_pos1 = self.pos_embedt(x1)
        x_pos1 = rearrange(x_pos1, 'b c h w -> b (h w) c')

        out_s = self.encoder_trans(x, x1, pos=x_pos)
        out_t = self.encoder_trant(x1, x, pos=x_pos1)

        outs1 = self.encoder_trans_1(ht, out_s, pos=None)
        outt1 = self.encoder_trant_1(ht, out_t, pos=None)

        ht_1 = self.juzheng(outs1, outt1, ht)
        ht_1 = ht_1 * 0.1 + ht

        outs2 = self.encoder_trans_2(outs1, outt1, pos=None)
        outt2 = self.encoder_trant_2(outt1, outs1, pos=None)

        outs = self.to_token(outs2)
        outt = self.to_tokent(outt2)
        out = torch.cat([outs, outt, ht_1], dim=1)
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dim_conv=2048, kernel=3, dropout=0.):
        """
        Encoder transformer block
        :param embed_dim: total dimension of the model
        :param num_heads: parallel attention heads
        :param dim_conv: feature in feedforward layer
        :param kernel: kernel size for feedforward operation, kernel=1 is similar to MLP layer
        :param dropout: a dropout layer on attention weight
        :param activation: activation function
        :param norm: normalization layer
        """
        super(TransformerEncoderLayer, self).__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.conv1 = nn.Conv2d(embed_dim, dim_conv, kernel_size=kernel, padding=int((kernel-1)/2))
        self.conv2 = nn.Conv2d(dim_conv, embed_dim, kernel_size=1, padding=0)

        self.norm1 = functools.partial(PixelwiseNorm)(embed_dim)
        self.norm2 = functools.partial(PixelwiseNorm)(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def _with_pos_embed(self, x, pos=None):
        return x if pos is None else x + pos

    def forward(self, src, src1, pos=None):
        b, c, h, w = src.size()
        src3 = self.norm1(src)
        src2 = self.norm1(src1)
        src2 = rearrange(src2, 'b c h w->b (h w) c')
        src3 = rearrange(src3, 'b c h w->b (h w) c')
        q = self._with_pos_embed(src3, pos)
        k = self._with_pos_embed(src2, pos)
        src2 = self.attn(q, k, src2)
        src2 = rearrange(src2, 'b (h w) c->b c h w', h=h, w=w)
        src = src + self.dropout(src2)
        src2 = self.norm2(src)
        src2 = self.conv2(self.dropout(self.activation(self.conv1(src2))))
        src = src + self.dropout(src2)
        return src

class TransformerEncoderLayer_1(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dim_conv=2048, kernel=3, dropout=0.):
        """
        Encoder transformer block
        :param embed_dim: total dimension of the model
        :param num_heads: parallel attention heads
        :param dim_conv: feature in feedforward layer
        :param kernel: kernel size for feedforward operation, kernel=1 is similar to MLP layer
        :param dropout: a dropout layer on attention weight
        :param activation: activation function
        :param norm: normalization layer
        """
        super(TransformerEncoderLayer_1, self).__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.conv1 = nn.Conv2d(embed_dim, dim_conv, kernel_size=kernel, padding=int((kernel-1)/2))
        self.conv2 = nn.Conv2d(dim_conv, embed_dim, kernel_size=1, padding=0)

        self.norm1 = functools.partial(PixelwiseNorm)(embed_dim)
        self.norm2 = functools.partial(PixelwiseNorm)(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def _with_pos_embed(self, x, pos=None):
        return x if pos is None else x + pos

    def forward(self, src, pos=None):
        b, c, h, w = src.size()
        src2 = self.norm1(src)
        src2 = rearrange(src2, 'b c h w->b (h w) c')
        q = k = self._with_pos_embed(src2, pos)
        src2 = self.attn(q, k, src2)
        src2 = rearrange(src2, 'b (h w) c->b c h w', h=h, w=w)
        src = src + self.dropout(src2)
        src2 = self.norm2(src)
        src2 = self.conv2(self.dropout(self.activation(self.conv1(src2))))
        src = src + self.dropout(src2)
        return src

class AttnAware(nn.Module):
    """Allows the model to jointly attend to information from different position"""

    def __init__(self, embed_dim=128, num_heads=4, dropout=0., bias=True):
        super(AttnAware, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.bias = bias
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_out = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        if self.bias:
            nn.init.constant_(self.to_q.bias, 0.)
            nn.init.constant_(self.to_k.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self, q, k, v):
        h1 = self.head_dim
        b, c, h, w = q.size()
        # calculate similarity map
        q = rearrange(q, 'b c h w->b (h w) c')
        k = rearrange(k, 'b c h w->b (h w) c')
        v = rearrange(v, 'b c h w->b (h w) c')
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q = rearrange(q, 'b n (h d)->b h n d', h=h1)
        k = rearrange(k, 'b n (h d)->b h n d', h=h1)
        v = rearrange(v, 'b n (h d)->b h n d', h=h1)
        dots = torch.einsum('bhid,bhjd->bhij', q, k)
        # calculate the attention value
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        # projection
        out = torch.einsum('bhij, bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'b (h w) c->b c h w', h=h, w=w)

        return out


class Auto_Attn(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(Auto_Attn, self).__init__()
        self.input_nc = input_nc
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        self.query_conv = nn.Conv2d(input_nc, input_nc // 4, kernel_size=(1,1))
        self.k_conv = nn.Conv2d(input_nc, input_nc // 4, kernel_size=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.nonlinearity = nn.LeakyReLU(0.1)

        self.conv1 = SpectralNorm(nn.Conv2d(input_nc , input_nc, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(input_nc, input_nc, **kwargs))
        self.shortcut = SpectralNorm(nn.Conv2d(input_nc , input_nc, kernel_size=(1, 1), stride=(1, 1)))
        self.model = nn.Sequential(self.nonlinearity, self.conv1, self.nonlinearity, self.conv2)

    def forward(self, q, k, v):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        B, C, W, H = q.size()
        proj_query = self.query_conv(q).view(B, -1, W * H)  # B X (N)X C
        proj_key = self.k_conv(k).view(B, -1, W * H)  # B X C x (N)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = v.view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.model(out) + self.shortcut(out)

        return out

