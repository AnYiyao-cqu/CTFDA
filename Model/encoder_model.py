"""
yancy F. 2020/10/31
For revised DASMN.
"""

from abc import ABC

import torch
import torch.nn.modules as nn
import torch.nn.functional as F

from my_utils.metric_utils import Euclidean_Distance


device = torch.device('cuda:0')
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Conv_CHN = 64
K_SIZE = 3
PADDING = (K_SIZE - 1) // 2

def normalize_weight(x):
    if x.numel() > 0:
        min_val = x.min()
        max_val = x.max()
        x = (x - min_val) / (max_val - min_val)

    else:
        x = 0
        x = torch.tensor(x)
    return x.detach()

class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR
        """

        b, n, dim = features.size()
        assert (n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda() #返回一个对角线全是1的数组64，
        features = F.normalize(features, dim=2)
        contrast_features = torch.cat(torch.unbind(features, dim=1),dim=0)
        anchor = features[:, 0]
        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()
        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # Mean log-likelihood for positive
        sim_loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean() + 1e-6

        return sim_loss

class wSimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(wSimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, wei):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR
        """

        b, n, dim = features.size()
        assert (n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda() #返回一个对角线全是1的数组64，
        features = F.normalize(features, dim=2)
        contrast_features = torch.cat(torch.unbind(features, dim=1),dim=0)
        anchor = features[:, 0]
        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()
        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask


        wei = torch.max(wei, dim=1)[1]
        wei = normalize_weight(wei)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # Mean log-likelihood for positive
        sim_loss = - (wei*(mask * log_prob).sum(1) / mask.sum(1)).mean() + 1e-6

        return sim_loss

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=K_SIZE, padding=PADDING),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
    )


class Encoder(nn.Module, ABC):
    def __init__(self, in_chn=1, cb_num=8):
        super().__init__()
        print('The Convolution Channel: {}'.format(Conv_CHN))
        print('The Convolution Block: {}'.format(cb_num))
        # self.se1 = SELayer(Convolution_CHN)
        conv1 = conv_block(in_chn, Conv_CHN)
        conv_more = [conv_block(Conv_CHN, Conv_CHN) for i in range(cb_num - 1)]
        self.conv_blocks = nn.Sequential(conv1, *conv_more)

    def forward(self, x):
        feat = self.conv_blocks(x)
        feat = feat.reshape(x.shape[0], -1)
        return feat


class MetricNet(nn.Module, ABC):
    def __init__(self, way, ns, nq, vis, cb_num=8):
        super().__init__()
        self.chn = 1
        self.way = way
        self.ns = ns
        self.nq = nq
        self.vis = vis
        self.encoder = Encoder(cb_num=cb_num)
        self.sim_loss = SimCLRLoss(6)
        self.wsim_loss = wSimCLRLoss(6)
        # self.criterion = nn.CrossEntropyLoss()  # ([n, nc], n)

    def get_loss(self, net_out, target_id, z_proto, conloss, conloss_1):
        # method 1:
        # log_p_y = torch.log_softmax(net_out, dim=-1).reshape(self.way, self.nq, -1)  # [nc, nq, nc]
        # loss_val = -log_p_y.gather(dim=2, index=target_ids).squeeze(dim=-1).reshape(-1).mean()
        # y_hat = log_p_y.max(dim=2)[1]  # [nc, nq]

        # method 2:

        log_p_y = torch.log_softmax(net_out, dim=-1)  # [nc*nq, nc], probability.
        loss = F.nll_loss(log_p_y, target_id.reshape(-1))  # (N, nc), (N,)
        corre = self.get_correlation(z_proto)
        unifor = self.get_uniformity(z_proto)
        loss += conloss + conloss_1 + 0.001 * (corre+unifor)
        y_hat = torch.max(log_p_y, dim=1)[1]  # [nc*nq]
        acc = torch.eq(y_hat, target_id).float().mean()

        return loss, acc, y_hat, -log_p_y.reshape(self.way, self.nq, -1)

    def get_features(self, x):
        return self.encoder.forward(x)

    def get_correlation(self, x):
        zq = x
        n = zq.size(0)

        # 初始化 L_c 为 0.0，确保它是一个标量
        L_c = torch.tensor(0.0, requires_grad=True)

        # 创建掩码，排除 zq 中的零值
        mask = zq != 0  # 掩码为 True 的元素是非零元素

        # 双重循环遍历 zq 中的每对元素，考虑掩码
        for i in range(n):
            for j in range(n):
                if i != j and mask[i].all().item() and mask[j].all().item():  # 只计算非零元素
                    p_c_a = zq[i]
                    p_c_b = zq[j]
                    # 计算 p_c^a \log(p_c^b \odot (p_c^a)^{-1})
                    L_c = L_c + p_c_a * torch.log(p_c_b / p_c_a)

                # 对结果求平均，考虑掩码

        valid_pairs = mask.sum().item() * (mask.sum().item() - 1)  # 有效的元素对数
        if valid_pairs > 0:
            L_c = (L_c / valid_pairs).mean()
        else:
            L_c = torch.tensor(0.0, requires_grad=True)

        return L_c

    def get_uniformity(self, x, t=0.1):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().add(1e-8).log()

    def forward(self, xs, xq, sne_state=False):
        # target_ids [nc, nq]
        target_id = torch.arange(self.way).unsqueeze(1).repeat([1, self.nq])
        target_id = target_id.long().to(device)
        # =================
        xs = xs.reshape(self.way * self.ns, self.chn, -1)
        xq = xq.reshape(self.way * self.nq, self.chn, -1)
        zs, zq = self.get_features(xs), self.get_features(xq)  # (nc*ns, z_dim)
        z_proto = zs.reshape(self.way, self.ns, -1).mean(dim=1)  # (nc, z_dim)


        dist = Euclidean_Distance(zq, z_proto)  # [nc*ns, nc]

        contra_feature = torch.stack((zs, zs), 1)
        simclr_loss = self.sim_loss(contra_feature)

        log_pre_y = torch.log_softmax(-dist, dim=-1)
        y_pse = torch.max(log_pre_y, dim=1)[1]
        ones = torch.ones_like(y_pse)
        comparison_result = torch.eq(y_pse, target_id.reshape(-1))
        result = torch.where(comparison_result, -ones, ones)
        ce_wei = torch.log_softmax((-dist*result.unsqueeze(1)), dim=0)
        z_proto_repeated = z_proto.repeat(zq.size(0) // z_proto.size(0), 1)
        contra_feature_1 = torch.stack((zq, z_proto_repeated), 1)
        wsimclr_loss_1 = (z_proto.size(0) // zq.size(0))*self.wsim_loss(contra_feature_1,ce_wei)
        loss_val, acc_val, y_hat, label_distribution = self.get_loss(-dist, target_id.reshape(-1), z_proto, simclr_loss, wsimclr_loss_1)

        # if sne_state and self.ns > 1:
        #     self.draw_feature(zq, target_id, y_hat)
        #     self.draw_label(label_distribution, target_id)

        return loss_val, acc_val, zq, target_id.reshape(-1), y_hat  # {'loss': loss_val.item(), 'acc': acc_val.item()}


if __name__ == "__main__":
    e = Encoder(cb_num=8)
    data = torch.ones([12, 1, 1024], dtype=torch.float)
    print(e)
    print(e.forward(data).shape)

