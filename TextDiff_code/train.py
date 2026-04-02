import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import gc
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
from src.utils import setup_seed
from src.pixel_classifier import pixel_classifier, pixel_classifier_condseg
from src.feature_extractors import create_feature_extractor, collect_features, collect_features_condseg
from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from utils import read_text
from src.load_dataset import RandomGenerator, ValGenerator, Mixdataset
import shutil
from tqdm import tqdm
from scipy.ndimage.interpolation import zoom
import cv2
import numpy as np
# ===== 新增导入 =====
import timm
import timm.layers.pos_embed


def dev():
    if torch.cuda.is_available():
        return torch.device(f"cuda")
    return torch.device("cpu")


def logger_config(log_path):
    logger = logging.getLogger()
    logger.propagate = False
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return loss

    def forward(self, preds, target, weight=None, softmax=False):
        if softmax:
            preds = torch.softmax(preds, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert preds.size() == target.size()
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(preds[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice_loss.item())
            loss += dice_loss * weight[i]
        return loss / self.n_classes


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)
    return iou, dice


# ===== 新增：cosine蒸馏损失 =====
def cosine_loss(z: torch.Tensor, z_tilde: torch.Tensor) -> torch.Tensor:
    """
    z:       DINOv2教师特征  [B, N, 768], frozen
    z_tilde: 学生投影特征    [B, N, 768], 可训练
    """
    assert z.shape == z_tilde.shape
    z_tilde = F.normalize(z_tilde, dim=-1)
    z = F.normalize(z, dim=-1)
    loss = -(z * z_tilde).sum(dim=-1).mean(dim=-1)  # [B]
    loss = loss.mean()                                # scalar
    return loss

'''
# ===== 新增：加载DINOv2 =====
@torch.no_grad()
def load_dinov2(model_name: str, repo_dir: str, resolution: int = 256) -> nn.Module:
    encoder = torch.hub.load(repo_dir, model_name, source='local')
    del encoder.head
    patch_resolution = 16 * (resolution // 256)  # =16 when resolution=256
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data, [patch_resolution, patch_resolution]
    )
    encoder.head = nn.Identity()
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder
'''

@torch.no_grad()
def load_dinov3(model_name: str, repo_dir: str, weights_path: str) -> nn.Module:
    """
    加载DINOv3 ViT-B/16模型
    model_name:   'dinov3_vitb16'
    repo_dir:     本地克隆的dinov3仓库路径
    weights_path: 下载的权重文件路径
    """
    encoder = torch.hub.load(
        repo_dir,
        model_name,
        source='local',
        weights=weights_path   # ← DINOv3新增的必填参数
    )
    # 不需要resample pos_embed（DINOv3用RoPE，天然支持任意分辨率）
    # 不需要删除head（DINOv3默认head就是Identity）
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder

# ===== 新增：四层特征的Projector =====
class FeatureProjector(nn.Module):
    """
    将四层空间特征图各自投影到DINOv2的token空间 [B, 256, z_dim]
    extract_dims: [1024, 512, 256, 64]  对应 feats[0~3]
    """
    def __init__(self, extract_dims, projector_dim=2048, z_dim=768):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )
            for dim in extract_dims
        ])

    def forward(self, features: dict) -> list:
        """
        features: {0:[B,1024,16,16], 1:[B,512,32,32], 2:[B,256,64,64], 3:[B,64,128,128]}
        返回: list of [B, 256, 768] × 4
        """
        z_tildes = []
        for k, projector in enumerate(self.projectors):
            feat = features[k]  # [B, C, H, W]
            # 统一插值到 16×16，与DINOv2的patch数量N=256对齐
            feat = F.interpolate(feat, size=(16, 16), mode='bilinear', align_corners=False)
            B, C, H, W = feat.shape
            feat = feat.view(B, C, H * W).transpose(-2, -1)  # [B, 256, C]
            z_tilde = projector(feat)                          # [B, 256, 768]
            z_tildes.append(z_tilde)
        return z_tildes


def evaluation(args, model, extractor, valloader):
    device = dev()
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=device).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], generator=rnd_gen, device=device)
    else:
        noise = None

    preds, gts = [], []

    for idx in tqdm(list(range(len(valloader))), desc='val:'):
        sample = valloader.dataset[idx]
        img, label, text, name = sample['image'], sample['label'], sample['text'], sample['name']
        img = img.unsqueeze(0).to(device)
        text = text.unsqueeze(0).to(device)
        features = extractor(img, noise=noise)
        features = collect_features_condseg(features)

        for k, v in features.items():
            features[k] = features[k].to(text.device)
        with torch.no_grad():
            pred = model(features, text)
            pred = F.interpolate(pred, size=(args["image_size"], args["image_size"]), mode='bilinear', align_corners=False)
            assert pred.dim() == 4 and pred.shape[1] > 1
            pred_softmax = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred_softmax, dim=1)
            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)

        mask_dir = "datasets/KSeg/test/masks"
        mask_name = sample["name"] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"[Error] 无法读取 mask 文件: {mask_path}")
            continue

        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = cv2.resize(mask, (args["image_size"], args["image_size"]), interpolation=cv2.INTER_NEAREST)

        gts.append(np.array(mask))
        preds.append(pred)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    for pred, target in zip(preds, gts):
        iou, dice = iou_score(pred, target)
        iou_avg_meter.update(iou, target.shape[0])
        dice_avg_meter.update(dice, target.shape[0])

    return dice_avg_meter.avg, iou_avg_meter.avg


def main(args, extractor, data_loader):

    device = dev()
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=device).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], generator=rnd_gen, device=device)
    else:
        noise = None

    gc.collect()

    extract_dims = [1024, 512, 256, 64]
    classifier = pixel_classifier_condseg(extract_dims=extract_dims)
    classifier.init_weights()
    classifier = classifier.cuda()

    # ===== 新增：初始化DINOv2和Projector =====
    use_alignment = args.get('use_alignment', True)

    if use_alignment:
        vision_encoder = load_dinov3(
            model_name=args['vision_encoder_model'],
            repo_dir=args['dinov3_repo_dir'],
            weights_path=args['dinov3_weights_path']
        )
        vision_encoder = vision_encoder.to(device)

        projector = FeatureProjector(
            extract_dims=extract_dims,
            projector_dim=args.get('projector_dim', 2048),
            z_dim=args.get('z_dim', 768)
        )
        projector = projector.cuda()

        # ImageNet归一化，用于DINOv2输入
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        lam = args.get('lam', 0.5)
    # ===== 新增结束 =====

    criterion_cross_entro = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=2)

    # ===== 修改：optimizer加入projector参数 =====
    if use_alignment:
        optimizer = torch.optim.Adam(
            list(classifier.parameters()) + list(projector.parameters()),
            lr=1e-4
        )
    else:
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    # ===== 修改结束 =====

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[60], gamma=0.1)

    stats = {'best_dice': 0., 'best_iou': 0., 'best_epoch': 0, 'best_ckpt': None}

    for epoch in range(args['max_training']):
        classifier.train()
        if use_alignment:
            projector.train()

        for idx in tqdm(list(range(len(data_loader)))):
            sample = data_loader.dataset[idx]
            img, label, text, name = sample['image'], sample['label'], sample['text'], sample['name']
            img   = img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            text  = text.unsqueeze(0).to(device)

            # 特征提取（frozen）
            features = extractor(img, noise=noise)
            features = collect_features_condseg(features)
            for k in features:
                features[k] = features[k].to(device)

            # 分割损失
            y_pred  = classifier(features, textf=text)
            y_batch = label.type(torch.long)
            optimizer.zero_grad()
            loss = criterion_cross_entro(y_pred, y_batch)
            loss += 1.5 * criterion_dice(y_pred, y_batch)

            # ===== 新增：蒸馏损失 =====
            if use_alignment:
                # 1. 提取DINOv2教师特征（frozen，no_grad）
                with torch.no_grad():
                    # img范围[0,1] → ImageNet归一化 → 插值到224
                    img_for_dino = F.interpolate(img, size=224, mode='bicubic', align_corners=False)
                    img_for_dino = (img_for_dino - imagenet_mean) / imagenet_std
                    z = vision_encoder.forward_features(img_for_dino)["x_norm_patchtokens"]
                    # z: [B, 256, 768]

                # 2. 四层特征各自投影
                z_tildes = projector(features)
                # z_tildes: list of [B, 256, 768] × 4

                # 3. 每层分别计算cosine loss，求平均
                distill_loss = sum([cosine_loss(z, z_tilde) for z_tilde in z_tildes]) / 4.0

                loss += lam * distill_loss
            # ===== 新增结束 =====

            loss.backward()
            optimizer.step()

        scheduler.step()

        with torch.no_grad():
            eval_dice, eval_iou = evaluation(opts, classifier.eval(), extractor=extractor, valloader=val_loader)
            if eval_dice > stats['best_dice']:
                stats['best_dice'] = eval_dice
                stats['best_iou']  = eval_iou
                stats['best_epoch'] = epoch

                # ===== 修改：保存时同时保存projector =====
                if use_alignment:
                    stats['best_ckpt'] = {
                        'classifier': classifier.state_dict(),
                        'projector':  projector.state_dict(),
                    }
                else:
                    stats['best_ckpt'] = classifier.state_dict()
                # ===== 修改结束 =====

                model_path = os.path.join(args['exp_dir'], 'model_' + f'{epoch:02d}.pth')
                torch.save({'model_state_dict': stats['best_ckpt']}, model_path)
                model_path = os.path.join(args['exp_dir'], 'model_best.pth')
                torch.save({'model_state_dict': stats['best_ckpt']}, model_path)

            logger.info(f"Epoch {epoch:02d}: dice/iou= {eval_dice:.4f}/{eval_iou:.4f}")

    logger.info(
        f'final model saved to: {os.path.join(args["exp_dir"], "model_best.pth")} '
        f'\n best_epoch:{stats["best_epoch"]} '
        f'best_dice:{stats["best_dice"]:.4f} '
        f'best_iou:{stats["best_iou"]:.4f}'
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int, default=40)

    args = parser.parse_args()
    setup_seed(args.seed)
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))

    os.makedirs(opts['exp_dir'], exist_ok=True)
    opts['exp_dir'] = os.path.join(opts['exp_dir'], f'experiment-{len(os.listdir(opts["exp_dir"]))+ 1:02d}')
    os.makedirs(opts['exp_dir'], exist_ok=True)
    print('Experiment folder: %s' % (opts['exp_dir']))
    shutil.copy(args.exp, opts['exp_dir'])

    train_text = read_text(os.path.join(opts['training_path'], 'KSeg_train.xlsx'))
    train_tf = RandomGenerator(output_size=[opts['image_size'], opts['image_size']])
    train_dt = Mixdataset(dataset_path=opts['training_path'], row_text=train_text, joint_transform=train_tf)

    val_text = read_text(os.path.join(opts['validation_path'], 'KSeg_test.xlsx'))
    val_tf = ValGenerator(output_size=[opts['image_size'], opts['image_size']])
    val_dt = Mixdataset(dataset_path=opts['validation_path'], row_text=val_text, joint_transform=val_tf)

    logger = logger_config(os.path.join(opts['exp_dir'], 'train.log'))
    train_loader = DataLoader(dataset=train_dt, batch_size=opts['batch_size'], shuffle=True,  drop_last=True)
    val_loader   = DataLoader(dataset=val_dt,   batch_size=opts['batch_size'], shuffle=False, drop_last=True)

    fea_extractor = create_feature_extractor(**opts)

    main(opts, extractor=fea_extractor, data_loader=train_loader)