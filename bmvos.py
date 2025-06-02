import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


# pre-, post-processing modules
def aggregate_objects(pred_seg, object_ids):
    bg_seg, _ = torch.stack([seg[:, 0, :, :] for seg in pred_seg.values()], dim=1).min(dim=1)
    bg_seg = torch.stack([1 - bg_seg, bg_seg], dim=1)
    logit = {n: seg[:, 1:, :, :].clamp(1e-7, 1 - 1e-7) / seg[:, 0, :, :].clamp(1e-7, 1 - 1e-7)
             for n, seg in [(-1, bg_seg)] + list(pred_seg.items())}
    logit_sum = torch.cat(list(logit.values()), dim=1).sum(dim=1, keepdim=True)
    aggregated_lst = [logit[n] / logit_sum for n in [-1] + object_ids]
    aggregated_inv_lst = [1 - elem for elem in aggregated_lst]
    aggregated = torch.cat([elem for lst in zip(aggregated_inv_lst, aggregated_lst) for elem in lst], dim=-3)
    mask_tmp = aggregated[:, 1::2, :, :].argmax(dim=-3, keepdim=True)
    pred_mask = torch.zeros_like(mask_tmp)
    for idx, obj_idx in enumerate(object_ids):
        pred_mask[mask_tmp == (idx + 1)] = obj_idx
    return pred_mask, {obj_idx: aggregated[:, 2 * (idx + 1):2 * (idx + 2), :, :] for idx, obj_idx in enumerate(object_ids)}


def get_padding(h, w, div):
    h_pad = (div - h % div) % div
    w_pad = (div - w % div) % div
    padding = [(w_pad + 1) // 2, w_pad // 2, (h_pad + 1) // 2, h_pad // 2]
    return padding


def attach_padding(imgs, given_masks, padding):
    B, L, C, H, W = imgs.size()
    imgs = imgs.view(B * L, C, H, W)
    imgs = F.pad(imgs, padding, mode='reflect')
    _, _, height, width = imgs.size()
    imgs = imgs.view(B, L, C, height, width)
    given_masks = [F.pad(label.float(), padding, mode='reflect').long() if label is not None else None for label in given_masks]
    return imgs, given_masks


def detach_padding(output, padding):
    if isinstance(output, list):
        return [detach_padding(x, padding) for x in output]
    else:
        _, _, _, height, width = output.size()
        return output[:, :, :, padding[2]:height - padding[3], padding[0]:width - padding[1]]


def add_coords(x):
    x_dim, y_dim = x.size(3), x.size(2)
    x_ones = torch.ones(1, x_dim).cuda()
    y_ones = torch.ones(y_dim, 1).cuda()
    x_range = torch.arange(y_dim).unsqueeze(1).float().cuda()
    y_range = torch.arange(x_dim).unsqueeze(0).float().cuda()
    x_ch = torch.matmul(x_range, x_ones).unsqueeze(0)
    y_ch = torch.matmul(y_ones, y_range).unsqueeze(0)
    x_ch = (x_ch / (y_dim - 1)) * 2 - 1
    y_ch = (y_ch / (x_dim - 1)) * 2 - 1
    x_ch = x_ch.repeat(x.size(0), 1, 1, 1)
    y_ch = y_ch.repeat(x.size(0), 1, 1, 1)
    return torch.cat([x, x_ch, y_ch, (x_ch ** 2 + y_ch ** 2) ** 0.5], dim=1)


# basic modules
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DeConv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('deconv', nn.ConvTranspose2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# encoding module
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = tv.models.densenet121(pretrained=True).features
        self.conv0 = backbone.conv0
        self.norm0 = backbone.norm0
        self.relu0 = backbone.relu0
        self.pool0 = backbone.pool0
        self.denseblock1 = backbone.denseblock1
        self.transition1 = backbone.transition1
        self.denseblock2 = backbone.denseblock2
        self.transition2 = backbone.transition2
        self.denseblock3 = backbone.denseblock3
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):
        x = (img - self.mean) / self.std
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.denseblock1(x)
        s4 = x
        x = self.transition1(x)
        x = self.denseblock2(x)
        s8 = x
        x = self.transition2(x)
        x = self.denseblock3(x)
        s16 = x
        return {'s4': s4, 's8': s8, 's16': s16}


# matching module
class Matcher(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = Conv(in_c, out_c, 1, 1, 0)

    def get_key(self, x):
        x = self.conv(x)
        key = x / x.norm(dim=1, keepdim=True)
        return key

    def forward(self, global_sim, local_sim, state):
        B, _, H, W = global_sim.size()

        # global matching
        score = global_sim * state['init_seg_16'][:, 0].view(B, H * W, 1, 1)
        bg_score = torch.max(score, dim=1, keepdim=True)[0]
        score = global_sim * state['init_seg_16'][:, 1].view(B, H * W, 1, 1)
        fg_score = torch.max(score, dim=1, keepdim=True)[0]
        global_score = torch.cat([bg_score, fg_score], dim=1)

        # local matching
        K = 4
        score = local_sim * state['prev_seg_16'][:, 0].view(B, H * W, 1, 1)
        score = score.view(B, H * W, H * W)
        topk = torch.topk(score, k=K, dim=2, sorted=True)[0]
        cut = topk[:, :, -1:].repeat(1, 1, H * W)
        min = torch.min(score, dim=2, keepdim=True)[0].repeat(1, 1, H * W)
        score[score < cut] = min[score < cut]
        score = score.view(B, H * W, H, W)
        bg_score = torch.max(score, dim=1, keepdim=True)[0]
        score = local_sim * state['prev_seg_16'][:, 1].view(B, H * W, 1, 1)
        score = score.view(B, H * W, H * W)
        topk = torch.topk(score, k=K, dim=2, sorted=True)[0]
        cut = topk[:, :, -1:].repeat(1, 1, H * W)
        min = torch.min(score, dim=2, keepdim=True)[0].repeat(1, 1, H * W)
        score[score < cut] = min[score < cut]
        score = score.view(B, H * W, H, W)
        fg_score = torch.max(score, dim=1, keepdim=True)[0]
        local_score = torch.cat([bg_score, fg_score], dim=1)

        # collect matching scores
        matching_score = torch.cat([global_score, local_score], dim=1)
        return matching_score


# mask embedding module
class MaskEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvRelu(9, 32, 7, 2, 3)
        self.conv2 = ConvRelu(32, 64, 7, 2, 3)
        self.conv3 = ConvRelu(64, 128, 7, 2, 3)
        self.conv4 = Conv(128, 256, 7, 2, 3)

    def forward(self, prev_segs):
        x = torch.cat([prev_segs[-1], prev_segs[-2], prev_segs[-3]], dim=1)
        mask_feats = self.conv4(self.conv3(self.conv2(self.conv1(add_coords(x)))))
        return mask_feats


# decoding module
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvRelu(1024, 256, 1, 1, 0)
        self.blend1 = ConvRelu(256 + 4 + 256, 256, 3, 1, 1)
        self.deconv1 = DeConv(256, 2, 4, 2, 1)
        self.conv2 = ConvRelu(512, 256, 1, 1, 0)
        self.blend2 = ConvRelu(256 + 2, 256, 3, 1, 1)
        self.deconv2 = DeConv(256, 2, 4, 2, 1)
        self.conv3 = ConvRelu(256, 256, 1, 1, 0)
        self.blend3 = ConvRelu(256 + 2, 256, 3, 1, 1)
        self.deconv3 = DeConv(256, 2, 6, 4, 1)

    def forward(self, feats, matching_score, mask_feats):
        x = torch.cat([self.conv1(feats['s16']), matching_score, mask_feats], dim=1)
        s8 = self.deconv1(self.blend1(x))
        x = torch.cat([self.conv2(feats['s8']), s8], dim=1)
        s4 = self.deconv2(self.blend2(x))
        x = torch.cat([self.conv3(feats['s4']), s4], dim=1)
        final_score = self.deconv3(self.blend3(x))
        return final_score


# VOS model
class VOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.matcher = Matcher(1024, 512)
        self.mask_embedder = MaskEmbedder()
        self.decoder = Decoder()

    def get_init_state(self, key, given_seg):

        # get init and prev segs
        state = {}
        given_seg_16 = F.avg_pool2d(given_seg, 16)
        state['init_seg_16'] = given_seg_16
        state['prev_seg_16'] = given_seg_16

        # get init and prev keys
        state['init_key'] = key
        state['prev_key'] = key

        # get mask feats
        state['prev_segs'] = [given_seg, given_seg, given_seg]
        state['mask_feats'] = self.mask_embedder(state['prev_segs'])
        return state

    def update_state(self, key, pred_seg, state):

        # update prev seg
        pred_seg_16 = F.avg_pool2d(pred_seg, 16)
        state['prev_seg_16'] = pred_seg_16

        # update prev key
        state['prev_key'] = key

        # update mask feats
        del state['prev_segs'][0]
        state['prev_segs'].append(pred_seg)
        state['mask_feats'] = self.mask_embedder(state['prev_segs'])
        return state

    def forward(self, feats, key, state):
        B, _, H, W = key.size()

        # get sim matrix
        init_key = state['init_key'].view(B, -1, H * W).transpose(1, 2)
        prev_key = state['prev_key'].view(B, -1, H * W).transpose(1, 2)
        global_sim = (torch.bmm(init_key, key.view(B, -1, H * W)).view(B, H * W, H, W) + 1) / 2
        local_sim = (torch.bmm(prev_key, key.view(B, -1, H * W)).view(B, H * W, H, W) + 1) / 2

        # get final score
        matching_score = self.matcher(global_sim, local_sim, state)
        final_score = self.decoder(feats, matching_score, state['mask_feats'])
        return final_score


# BMVOS model
class BMVOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.vos = VOS()

    def forward(self, imgs, given_masks, val_frame_ids):

        # basic setting
        B, L, _, H, W = imgs.size()
        padding = get_padding(H, W, 16)
        if tuple(padding) != (0, 0, 0, 0):
            imgs, given_masks = attach_padding(imgs, given_masks, padding)

        # initial frame
        object_ids = given_masks[0].unique().tolist()
        if 0 in object_ids:
            object_ids.remove(0)
        mask_lst = [given_masks[0]]

        # initial frame embedding
        feats = self.vos.encoder(imgs[:, 0])
        key = self.vos.matcher.get_key(feats['s16'])

        # create state for each object
        state = {}
        for k in object_ids:
            given_seg = torch.cat([given_masks[0] != k, given_masks[0] == k], dim=1).float()
            state[k] = self.vos.get_init_state(key, given_seg)

        # subsequent frames
        for i in range(1, L):

            # query frame embedding
            feats = self.vos.encoder(imgs[:, i])
            key = self.vos.matcher.get_key(feats['s16'])

            # query frame prediction
            pred_seg = {}
            for k in object_ids:
                final_score = self.vos(feats, key, state[k])
                pred_seg[k] = torch.softmax(final_score, dim=1)

            # detect new object
            if given_masks[i] is not None:
                new_object_ids = given_masks[i].unique().tolist()
                if 0 in new_object_ids:
                    new_object_ids.remove(0)
                for new_k in new_object_ids:
                    given_seg = torch.cat([given_masks[i] != new_k, given_masks[i] == new_k], dim=1).float()
                    state[new_k] = self.vos.get_init_state(key, given_seg)
                    pred_seg[new_k] = torch.cat([given_masks[i] != new_k, given_masks[i] == new_k], dim=1).float()
                object_ids = object_ids + new_object_ids

            # aggregate objects
            pred_mask, pred_seg = aggregate_objects(pred_seg, object_ids)

            # update state
            if i < L - 1:
                for k in object_ids:
                    state[k] = self.vos.update_state(key, pred_seg[k], state[k])

            # store hard masks
            if given_masks[i] is not None:
                pred_mask[given_masks[i] != 0] = 0
                mask_lst.append(pred_mask + given_masks[i])
            else:
                if val_frame_ids is not None:
                    if val_frame_ids[0] + i in val_frame_ids:
                        mask_lst.append(pred_mask)
                else:
                    mask_lst.append(pred_mask)

        # generate output
        output = {}
        output['masks'] = torch.stack(mask_lst, dim=1)
        output['masks'] = detach_padding(output['masks'], padding)
        return output
