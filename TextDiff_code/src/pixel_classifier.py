import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from typing import Optional
from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from src.transformer import BasicTransformerBlock

# criterion_KL = nn.KLDivLoss(reduction='batchmean')



################
class MSAtt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_spatial_1 = nn.Conv2d(dim, dim, 3, stride=1, padding=6, groups=dim, dilation=6)
        self.conv_spatial_2 = nn.Conv2d(dim, dim, 3, stride=1, padding=12, groups=dim, dilation=12)
        self.conv_spatial_3 = nn.Conv2d(dim, dim, 3, stride=1, padding=18, groups=dim, dilation=18)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn_1 = self.conv_spatial_1(x)
        attn_2 = self.conv_spatial_2(x)
        attn_3 = self.conv_spatial_3(x)
        attn = attn_1 + attn_2 + attn_3
        attn = self.conv1(attn)

        return u * attn

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True, groups=in_channels)
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dwconv(x)
        weights = self.conv1(x)
        weights = self.sigmoid(weights)
        attended_x = x * weights
        return attended_x

class MSM(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.ReLU(inplace=True)
        self.dwspatialatt=SpatialAttention(d_model)
        self.spatial_gating_unit = MSAtt(d_model)
        self.proj_2 = nn.Conv2d(d_model*2, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        xdw = self.dwspatialatt(x)
        xsg = self.spatial_gating_unit(x)
        x = self.proj_2(torch.cat([xdw,xsg],dim=1))
        x = x + shorcut
        return x



#################

class pixel_classifier_condseg(nn.Module):
    def __init__(self, extract_dims):
        super(pixel_classifier_condseg, self).__init__()
        self.extract_dims = extract_dims

        self.decodes = nn.Sequential(
            nn.Conv2d(3072, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, 1, 1),
        )

        self.cross1 = BasicTransformerBlock(dim=768, num_attention_heads=8, attention_head_dim=64, dropout=0.0, cross_attention_dim=768,
                                            activation_fn='geglu', num_embeds_ada_norm=None, attention_bias=False, only_cross_attention=True)
        
        
        self.MSM1 = MSM(extract_dims[0])
        self.MSM2 = MSM(extract_dims[1])
        self.MSM3 = MSM(extract_dims[2])
        self.MSM4 = MSM(extract_dims[3])
        
        self.down1 = nn.Conv2d(extract_dims[0], 768, 1, 1, 0)
        self.down2 = nn.Conv2d(extract_dims[1], 768, 1, 1, 0)
        self.down3 = nn.Conv2d(extract_dims[2], 768, 1, 1, 0)
        self.down4 = nn.Conv2d(extract_dims[3], 768, 1, 1, 0)
        
        

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, imgfs, textf):
        
        feats = []
        for k, (idx, imgf) in enumerate(imgfs.items()):
            imgf = getattr(self, f'MSM{idx+1}')(imgf)
            
            imgf = imgf if imgf.shape[1] == 768 else getattr(self, f'down{idx+1}')(imgf)
            
            
            
            imgf = imgf.flatten(2).permute(0, 2, 1)
            fusionf = self.cross1(imgf, textf)
            feats.append(fusionf)
        
        
        aggs = []
        for f in feats:
            s = int(np.sqrt(f.shape[1]))
            f = torch.nn.functional.interpolate(f.permute(0, 2, 1).reshape(1, -1, s, s), size=(256, 256), mode='bilinear')
            aggs.append(f)
        aggs = torch.cat(aggs, dim=1)
        out = self.decodes(aggs)

        return out

class pixel_classifier(nn.Module):
    def __init__(self, extract_dims):
        super(pixel_classifier, self).__init__()
        self.extract_dims = extract_dims

        self.decodes = nn.Sequential(
            nn.Conv2d(3072, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, 1, 1),
        )

        self.cross1 = BasicTransformerBlock(dim=768, num_attention_heads=8, attention_head_dim=64, dropout=0.0, cross_attention_dim=768,
                                            activation_fn='geglu', num_embeds_ada_norm=None, attention_bias=False, only_cross_attention=True)
        self.down1 = nn.Conv2d(extract_dims[0], 768, 1, 1, 0)
        self.down2 = nn.Conv2d(extract_dims[1], 768, 1, 1, 0)

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, imgfs, textf):
        
        feats = []
        for k, (idx, imgf) in enumerate(imgfs.items()):
            imgf = imgf if imgf.shape[1] == 768 else getattr(self, f'down{min(idx+1,2)}')(imgf)
            imgf = imgf.flatten(2).permute(0, 2, 1)
            fusionf = self.cross1(imgf, textf)
            feats.append(fusionf)
            
        aggs = []
        for f in feats:
            s = int(np.sqrt(f.shape[1]))
            f = torch.nn.functional.interpolate(f.permute(0, 2, 1).reshape(1, -1, s, s), size=(256, 256), mode='bilinear')
            aggs.append(f)
        aggs = torch.cat(aggs, dim=1)
        out = self.decodes(aggs)

        return out



def predict_labels(models, features, text, size):
    # if isinstance(features, np.ndarray):
    #     features = torch.from_numpy(features)

    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            model = models[MODEL_NUMBER]
            preds, *_ = model(features, text)
            preds = preds[None]
            # preds = model(features.cuda())
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]
    # return img_seg_final, top_k
    return seg_mode_ensemble[0], top_k


def compute_iou(args, preds, gts, print_per_class_ious=True):
    ids = range(args['number_class'])
    unions = Counter()
    intersections = Counter()
    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']:
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
    ious = []
    for target_num in ids:
        if target_num == args['ignore_label']:
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)

    return np.array(ious).mean()


def load_ensemble(args, device='cpu'):
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']
        # model = nn.DataParallel(pixel_classifier(args["number_class"], args['dim'][-1]))
        model = pixel_classifier(args["number_class"], args['dim'][0])
        model.load_state_dict(state_dict)
        # model = model.module.to(device)
        model = model.to(device)
        models.append(model.eval())
    return models
