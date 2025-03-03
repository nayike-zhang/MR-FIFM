# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import math
import operator
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from layers.triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from layers.weighted_dist_cp import WeightedTripletLoss, ParamTripletLoss
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP, Weighted_R1_mAP, Param_R1_mAP
from utils.mytools import ResultSaver
from bayes_opt import BayesianOptimization
from functools import reduce
from utils.mytools import LOSS_ZERO

from ray import tune

from layers.triplet_loss import euclidean_dist

from layers import fastreid_circle_loss
from utils.mytools import modal_rand_missing

def normalize(feat, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      feat: pytorch Variable
    Returns:
      feat: pytorch Variable, same shape as input
    """
    feat = 1. * feat / (torch.norm(feat, 2, axis, keepdim=True).expand_as(feat) + 1e-12)
    return feat

def hetero_loss(f_list, label, margin=0.1): # CdC-M loss
    uni_label_num = len(label.unique())
    chunk_f_list = [f.chunk(uni_label_num, 0) for f in f_list]
    l = len(f_list)

    dist = 0
    dist_func = nn.MSELoss(reduction='sum')
    center_list = []
    for l_idx in range(uni_label_num):
        for f in chunk_f_list:
            center_list.append(torch.mean(f[l_idx], 0))
        for i in range(l):
            for j in range(i+1, l):
                dist += max(0, dist_func(center_list[i], center_list[j]) - margin)

    return 2*dist/(l*(l-1))

def MMIC_LOSS(f_list, label, margin): # CdC-S loss
    uni_label_num = len(label.unique())
    chunk_f_list = [f.chunk(uni_label_num, 0) for f in f_list]
    modalities = len(f_list)

    dist = 0
    for i in range(uni_label_num):
        f = [chunk_f_list[n][i] for n in range(modalities)]
        # import pdb; pdb.set_trace()
        dist += MultiModalIdConsistLoss(f, margin=margin)
    return dist/uni_label_num

def MultiModalIdConsistLoss(f_list, margin=0.1):
    # feat should belong to same class
    N_f = f_list[0].shape[0]
    N_m = len(f_list)

    center_list = []
    for i in range(N_f):
        center_f = 0
        for n in range(N_m):
            center_f += f_list[n][i]
        center_f = center_f/N_m
        center_list.append(center_f)

    l = len(center_list)
    assert l >= 2
    dist_func = nn.MSELoss(reduction='sum')
    dist = 0
    for i in range(l):
        for j in range(i+1, l):
            dist += max(0, dist_func(center_list[i], center_list[j]) - margin)

    return 2*dist/(l*(l-1))


trip_loss = TripletLoss(margin=0.3)
def MultiModalTriplet(f_list, label): # 样本中心三元组损失
    modalities = len(f_list)
    modal_center_feats = sum(f_list)/modalities
    loss = trip_loss(modal_center_feats, label)[0]
    return loss

class CMDLoss(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMDLoss, self).__init__()

    def forward(self, x1, x2, n_moments):
        x1 = F.normalize(x1, dim=1)  # (bs, dim)  --->  (bs, dim)
        #x1 = torch.sigmoid(x1)
        x2 = F.normalize(x2, dim=1)  # (bs, dim)  --->  (bs, dim)
        #x2 = torch.sigmoid(x2)

        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)

        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2, 2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_i = torch.sigmoid(z_i)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = torch.sigmoid(z_j)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss



# from light-reid
def kl_div_loss(logits_s, logits_t, mini=1e-8): # KL散度
    '''
    :param logits_s: student score
    :param logits_t: teacher score as target
    :param mini: for number stable
    :return:
    '''
    logits_t = logits_t.detach()
    prob1 = F.softmax(logits_s, dim=1)
    prob2 = F.softmax(logits_t, dim=1)
    loss = torch.sum(prob2 * torch.log(mini + prob2 / (prob1 + mini)), 1) + torch.sum(prob1 * torch.log(mini + prob1 / (prob2 + mini)), 1)
    return loss.mean()


def ClassCompactLoss(feat, alpha=1.5, margin=0.2, feat_norm=False): # 关于类内聚的尝试
    # feat should belong to same class
    if feat_norm:
        feat = 1. * feat / (torch.norm(feat, 2, axis=-1, keepdim=True).expand_as(feat) + 1e-12)
    B = feat.shape[0]
    assert B > 1
    center = torch.mean(feat, dim=0)
    dist = torch.pow(feat - center, 2)
    dist = torch.sum(dist)/(B - 1)
    # dist = torch.sum(torch.pow(torch.std(feat, dim=0), 2))
    dist = max(dist - margin, 0)
    loss = torch.exp(alpha*dist) - 1
    return loss, center


class AngleLoss(nn.Module):#设置loss，超参数gamma，最小比例，和最大比例
    def __init__(self, gamma=0, lambda_min=5, lambda_max=1500):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, x, y): #分别是output和target
        self.it += 1
        cos_theta, phi_theta = x #output包括上面的[cos_theta, phi_theta]
        y = y.view(-1, 1)

        index = cos_theta.data * 0.0
        index.scatter_(1, y.data.view(-1, 1), 1)#将label存成稀疏矩阵
        index = index.byte()
        index = Variable(index)

        lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.it))#动态调整lambda，来调整cos(\theta)和\phi(\theta)的比例
        output = cos_theta * 1.0
        output[index] -= cos_theta[index]*(1.0+0)/(1 + lamb)#减去目标\cos(\theta)的部分
        output[index] += phi_theta[index]*(1.0+0)/(1 + lamb)#加上目标\phi(\theta)的部分

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, y)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

def CenterMarginLoss(f_list, label, margin, engine): # 关于类可分的尝试
    uni_label = label.unique()
    uni_label_num = len(uni_label)
    chunk_f_list = [f.chunk(uni_label_num, 0) for f in f_list]
    l = len(f_list)

    all_feat = torch.cat(f_list, dim=0)
    all_label = torch.cat([label]*l)

    final_loss = 0
    for i in range(uni_label_num):
        feat = torch.cat([chunk_f_list[j][i] for j in range(l)], dim=0)
        center_feat = torch.mean(feat, dim=0, keepdim=True)
        dist_mat = euclidean_dist(center_feat, all_feat).squeeze()
        is_pos = all_label.eq(uni_label[i])
        is_neg = 1 - is_pos

        max_pos_dist = torch.max(dist_mat[is_pos])
        min_neg_dist = torch.min(dist_mat[is_neg])

        if engine.state.epoch > 805:
            import pdb; pdb.set_trace()

        final_loss += max(0, margin + max_pos_dist - min_neg_dist)
    return final_loss

def rand_missing_loss(feats, margin=0.05, missing_rate=0.3): # 关于随机缺失损失的尝试
    N = feats[0].shape[0]
    modalities = len(feats)
    rand = torch.rand(size=(N, modalities)).cuda()
    miss_mask = rand < missing_rate

    mask_sum = torch.sum(miss_mask, dim=1)
    keep_mask = mask_sum.eq(modalities)
    miss_mask = miss_mask - keep_mask.unsqueeze(dim=1)
    mask = 1 - miss_mask
    mask_sum = torch.sum(mask, dim=1).unsqueeze(dim=1)

    feats = torch.stack(feats, dim=1) # (B, 3, 2048)
    keep_feats = feats * mask.unsqueeze(dim=2).float()

    c1 = torch.sum(feats, dim=1) / modalities
    c2 = torch.sum(keep_feats, dim=1) / mask_sum.float()

    loss = nn.MSELoss(reduction='mean')(c1, c2)
    loss = max(0, loss - margin)

    return loss

def rand_missing_loss_2(feats, label, margin=0.05, missing_rate=0.3): # 关于随机缺失损失的尝试
    N = feats[0].shape[0]
    modalities = len(feats)
    rand = torch.rand(size=(N, modalities)).cuda()
    miss_mask = rand < missing_rate

    mask_sum = torch.sum(miss_mask, dim=1)
    keep_mask = mask_sum.eq(modalities)
    miss_mask = miss_mask - keep_mask.unsqueeze(dim=1)
    mask = 1 - miss_mask
    mask = mask.unsqueeze(dim=2).float() # (B, 3, 1)
    feats = torch.stack(feats, dim=1) # (B, 3, 2048)
    keep_feats = feats * mask

    dist_func = nn.MSELoss(reduction='mean')
    id_num = len(label.unique())
    id_samples = N / id_num
    id_feats = feats.chunk(id_num, dim=0)
    id_keep_feats = keep_feats.chunk(id_num, dim=0)
    id_mask = mask.chunk(id_num, dim=0)
    loss = 0
    for i in range(id_num):
        # missing consistency in samples
        c = id_feats[i].sum(dim=0) / id_samples
        c_m = id_keep_feats[i].sum(dim=0) / id_mask[i].sum(dim=0, keepdim=True)
        loss += max(0, dist_func(c, c_m) - margin)

        # missing consistency in modal
        # c = id_feats[i].sum(dim=0) / id_samples
        # c_m = id_keep_feats[i].sum(dim=0) / id_mask[i].sum(dim=0, keepdim=True)
        # loss += max(0, dist_func(c, c_m) - margin)

    return loss

def r_trans(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.item()
    else:
        return tensor

# 训练核心代码
def create_supervised_trainer(cfg, model, optimizer, loss_fn, device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    trip_loss = TripletLoss(margin=0.3)
    # w_trip_loss = WeightedTripletLoss(margin=0.3)
    # from layers.myloss import ModifiedTripletLoss as CMG
    # trip_loss = CMG(margin=0.3)
    # circle_loss = CircleLoss(m=0.25, gamma=256)
    # p_trip_loss = ParamTripletLoss(3, 0.3)
    # optimizer_ptrip = torch.optim.SGD(p_trip_loss.parameters(), lr=0.5)
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        # optimizer_ptrip.zero_grad()

        imgs, target, _, _, tain_path = batch  # 传入每一批次数据，包括8个身份，每个身份4个样本
        # print(target)
        # print("tain_path", tain_path)
        # imgs, mask = modal_rand_missing([img1, img2, img3], prob=0.1)
        # img1, img2, img3 = imgs
        for i in range(len(imgs)):
            imgs[i] = imgs[i].to(device) if torch.cuda.device_count() >= 1 else imgs[i]
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        B = target.shape[0]
        if cfg.MODEL.BRANCHES == 3:
            pred_s_list, pred_p_list, s_list, p_list, weight_list, bn_s_list, bn_p_list, fuse_pred_list, fuse_feat = model(imgs[:3])
        elif cfg.MODEL.BRANCHES == 2:
            pred_s_list, pred_p_list, s_list, p_list, weight_list, bn_s_list, bn_p_list, fuse_pred_list, fuse_feat = model(imgs[:2])
        else:
            raise NotImplementedError

        #cls_loss_0 = 0.#+ trip_loss(gf_list[0], target)[0]

        #cls_loss_1 = 0.87*(trip_loss(gf_list[0], target)[0] + trip_loss(gf_list[1], target)[0] + trip_loss(fuse_feat, target)[0])

        #cls_loss_2 = 1.2*(kl_div_loss(fuse_pred_list[0], pred_list[0]) + kl_div_loss(fuse_pred_list[0], pred_list[1])) #+ trip_loss(fuse_feat, target)[0]
        # print(torch.sum(pred_))



        #cls_loss_0 = F.cross_entropy(pred_s_list[0], target) + F.cross_entropy(pred_s_list[1], target)
        #cls_loss_0 = F.cross_entropy(fuse_pred_list[0], target)
        #cls_loss_0 = trip_loss(fuse_feat, target)[0]

        #xent = CrossEntropyLabelSmooth(num_classes=300)
        criterion = nn.CosineEmbeddingLoss(margin=0.3)
        cls_loss_0 = F.cross_entropy(fuse_pred_list[0], target) #+ 1.22*(F.cross_entropy(pred_s_list[0], target) + F.cross_entropy(pred_s_list[1], target))

       # loss_func = ContrastiveLoss(batch_size=s_list[0].shape[0])

        loss_1 = CMDLoss()

        cls_loss_1 = 1.44*kl_div_loss(s_list[0], s_list[1]) #1.60*loss_1(s_list[0], s_list[1], 5)#0.6*loss_func(s_list[0], s_list[1])


        cls_loss_2 = 0.69*(trip_loss(fuse_feat, target)[0]+ trip_loss(s_list[0], target)[0] + trip_loss(s_list[1], target)[0]) # 0.0045466369379413 98.49

        cls_loss_3 = 0.#kl_div_loss(fuse_pred_list[0], pred_s_list[0]) + kl_div_loss(fuse_pred_list[0], pred_s_list[1])

        loss_list = [cls_loss_0, cls_loss_1, cls_loss_2, cls_loss_3]

        total_loss = sum(loss_list)
        total_loss.backward()
        optimizer.step()


        get_acc = lambda score, target: (score.max(1)[1] == target).float().mean()

        pp_acc = get_acc(pred_s_list[0], target)
        pv_acc = get_acc(pred_s_list[1], target)

        fuse_acc = get_acc(fuse_pred_list[0], target)

        # print("branch_acc[2]",branch_acc[2], fuse_pred_list[0].max(1)[1], target)
        return r_trans(total_loss), r_trans(cls_loss_0), r_trans(cls_loss_1), r_trans(cls_loss_2), r_trans(cls_loss_3), r_trans(pp_acc), r_trans(pv_acc), r_trans(fuse_acc)

    # return Engine(_update), p_trip_loss.weights
    return Engine(_update)

# 训练过程中途测试的代码
def create_evaluator(model, metrics, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
        return_ctler: function to decide content returned
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    cnt = 0

    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
    # total_acc = np.array([0,0,0,0])
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():

            imgs, target, camids, sceneids, img_path = batch
            # print("img_path", img_path)
            for i in range(len(imgs)):
                imgs[i] = imgs[i].to(device) if torch.cuda.device_count() >= 1 else imgs[i]
            #print("target",target)
            target = torch.tensor(np.array(target))
            target = target.to(device) if torch.cuda.device_count() >= 1 else target
            pred_s_list, pred_p_list, s_list, p_list, weight_list, bn_s_list, bn_p_list, fuse_pred_list, fuse_feat = model(imgs[:len(imgs)])

            get_acc = lambda score, target: (score.max(1)[1] == target).float().mean()

            pp_acc = get_acc(pred_s_list[0], target)
            pv_acc = get_acc(pred_s_list[1], target)

            fuse_acc = get_acc(fuse_pred_list[0], target)

            target = target.cpu()

            return target, camids, sceneids, img_path, pp_acc, pv_acc, fuse_acc, pred_s_list, fuse_pred_list[0].cpu()#.max(1)[1]#.cpu().numpy()

    engine = Engine(_inference)

    for name, metric in metrics.items():
        #print("name, metric", name, metric)
        metric.attach(engine, name)
    print("Test" * 10)
    #print(total_acc)
    #self.total_acc = np.array([0,0,0,0])
    return engine

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    #pdb.set_trace()

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    print("LAMBDA:", cfg.LAMBDA)
    print("ALPHA:", cfg.ALPHA)
    # trainer, p_weights = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    trainer = create_supervised_trainer(cfg, model, optimizer, loss_fn, device=device)

    metrics_r1map = {'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)} # 计算rank mAP
    metrics_wr1map = {'w_r1_mAP': Weighted_R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)} # 很久之前尝试的一红加权度量思路
    # metrics_pr1map = {'p_r1_mAP': Param_R1_mAP(num_query, max_rank=50, weight_param=p_weights, feat_norm=cfg.TEST.FEAT_NORM)}

    # 下列evaluator通过指定测试特征，比如三模态特征都用或者只用某一个或俩个，或者你想要的任何方式

    evaluator_bnf = create_evaluator(model, metrics=metrics_r1map, device=device)

    # p_evaluator_gf = create_evaluator(model, metrics=metrics_pr1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: g_fs)
    # p_evaluator_bnf = create_evaluator(model, metrics=metrics_pr1map, device=device, \
    #     return_ctler=lambda  g_fs, weights, bn_f_list: bn_f_list)


    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False, save_as_state_dict=False)

    timer = Timer(average=True)
    R_Saver = ResultSaver()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict()})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    # 顺序要跟create_supervised_trainer的返回值一致
    RunningAverage(output_transform=lambda feat: feat[0]).attach(trainer, 'total_loss')   # 总损失
    RunningAverage(output_transform=lambda feat: feat[1]).attach(trainer, 'cls_loss_0')    # 预测列表与目标acc
    RunningAverage(output_transform=lambda feat: feat[2]).attach(trainer, 'cls_loss_1')      # 交叉熵损失
    RunningAverage(output_transform=lambda feat: feat[3]).attach(trainer, 'cls_loss_2')
    RunningAverage(output_transform=lambda feat: feat[4]).attach(trainer, 'cls_loss_3')     # 三元组损失
    RunningAverage(output_transform=lambda feat: feat[5]).attach(trainer, 'pp_acc')
    RunningAverage(output_transform=lambda feat: feat[6]).attach(trainer, 'pv_acc')
    RunningAverage(output_transform=lambda feat: feat[7]).attach(trainer, 'fuse_acc')




    #下面的内容是指定在什么时刻该干什么，比如第一个，在Events.STARTED也就是训练开始时，做xxx

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader),
                                engine.state.metrics['total_loss'], engine.state.metrics['fuse_acc'],
                                scheduler.get_lr()[0]))
            # g = lambda name: engine.state.metrics[name]
            # logger.info('Loss=> cls_0:{:.3f} cls_1:{:.3f} cls_2:{:.3f} hc:{:.3f} mmic:{:.3f} mmtri:{:.3f}'.format(g('cls_0'), g('cls_1'), g('cls_2'),  g('loss_hc'), g('mmic_loss'), g('mmtri_loss')))



    # adding handlers using `trainer.on` decorator API
    # 在每个epoch结束时，按格式打印结果
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        g = lambda name: engine.state.metrics[name]
        path = "NUPT1/_cnnhu9_"
        logger.info('Loss=> cls_0:{:.3f} cls_1:{:.3f} cls_2:{:.3f} cls_3:{:.3f}'.format(g('cls_loss_0'), g('cls_loss_1'), g('cls_loss_2'), g('cls_loss_3')))
        logger.info('Acc=> pp_acc:{:.2%} pv_acc:{:.2%} fuse_acc:{:.2%} '.format(g('pp_acc'), g('pv_acc'), g('fuse_acc')))
        logger.info('-' * 10)
        timer.reset()
        f0 = open("/home/zhangli/CCNet/outputs/" + path + "loss0.txt","a")
        f1 = open("/home/zhangli/CCNet/outputs/" + path + "loss1.txt","a")
        f2 = open("/home/zhangli/CCNet/outputs/" + path + "loss2.txt","a")
        f0.write(str(g('cls_loss_0')) + ",")
        f1.write(str(g('cls_loss_1')) + ",")
        f2.write(str(g('cls_loss_2')) + ",")
        f0.flush()
        f0.close()
        f1.flush()
        f1.close()
        f2.flush()
        f2.close()
    # 在每个epoch结束时，看是否需要测试，如果需要则进行测试，并按格式打印结果
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:

            def __print_info(info, evaluator, metric_name):

                logger.info(' ')
                logger.info(info)
                evaluator.run(val_loader)
                path = "NUPT1/_cnnhu9_"
                acc = evaluator.state.metrics[metric_name]
                print("metric_name", metric_name)
                logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
                logger.info("pp_acc: {:.2%} pv_acc: {:.2%} acc_fuse: {:.2%}".format(acc[0], acc[1], acc[2]))

                f0 = open("/home/zhangli/CCNet/outputs/" + path +"acc.txt","a")
                f0.write(str(acc[2]) + ",")
                f0.flush()
                f0.close()
                return acc[2]

                #logger.info("mAP: {:.1%}".format(mAP))
                #for r in [1, 5, 10]:
                #    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                #R_Saver.add(info, (engine.state.epoch, mAP, cmc[0], cmc[4], cmc[9]))

            # __print_info('1. using final feature for test', evaluator, 'r1_mAP')
            ###########__print_info('a. all global feat concat', evaluator_gf, 'r1_mAP')
            # __print_info('b. rgb feat', evaluator_rgb, 'r1_mAP')
            # __print_info('c. nir feat', evaluator_ni, 'r1_mAP')
            # __print_info('d. tir feat', evaluator_t, 'r1_mAP')
            # __print_info('e. r_n feat', evaluator_rn, 'r1_mAP')
            # __print_info('f. r_t feat', evaluator_rt, 'r1_mAP')
            # __print_info('g. n_t feat', evaluator_nt, 'r1_mAP')
            # # __print_info('2. [1] - fusing feat', evaluator_gf_nofuse, 'r1_mAP')
            # __print_info('3. [1] + weighted metric', w_evaluator_gf, 'w_r1_mAP')
            # # __print_info('4. [2] + weighted metric', w_evaluator_gf_nofuse, 'w_r1_mAP')
            __print_info('h. all bn feat concat', evaluator_bnf, 'r1_mAP')
            # __print_info('i. bn rgb feat', evaluator_bn_rgb, 'r1_mAP')
            # __print_info('j. bn nir feat', evaluator_bn_ni, 'r1_mAP')
            # __print_info('k. bn tir feat', evaluator_bn_t, 'r1_mAP')
            # __print_info('l. bn r_n feat', evaluator_bn_rn, 'r1_mAP')
            # __print_info('m. bn r_t feat', evaluator_bn_rt, 'r1_mAP')
            # __print_info('n. bn n_t feat', evaluator_bn_nt, 'r1_mAP')

            # __print_info('6. [5] - fusing feat', evaluator_bnf_nofuse, 'r1_mAP')
            # __print_info('7. [5] + weighted metirc', w_evaluator_bnf, 'w_r1_mAP')
            # __print_info('8. [6] + weighted metric', w_evaluator_bnf_nofuse, 'w_r1_mAP')
            # __print_info('9. p_weighted, gf', p_evaluator_gf, 'p_r1_mAP')
            # __print_info('10. p_weighted, bnf', p_evaluator_bnf, 'p_r1_mAP')
            # normed_p_weights = p_weights.detach().cpu().numpy()
            # # min_p = min(normed_p_weights)
            # # max_p = max(normed_p_weights)
            # # normed_p_weights = (normed_p_weights - min_p)/(max_p - min_p + 1e-12)
            # normed_p_weights = np.exp(normed_p_weights)
            # normed_p_weights = normed_p_weights / np.sum(normed_p_weights)
            # logger.info('parameters in p_trip:\n {}'.format(normed_p_weights.reshape((3, 3))))
            R_Saver.saveResults(cfg.OUTPUT_DIR)

    trainer.run(train_loader, max_epochs=epochs)
    # R_Saver.saveAsCSV(cfg.OUTPUT_DIR)

def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    pass