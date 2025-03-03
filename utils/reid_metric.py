# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from layers.triplet_loss import normalize
import numpy as np
import torch
import pdb
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
import torch.nn.functional as F
from layers.weighted_dist_cp import weighted_dist_comp
from layers.weighted_dist_cp import param_dist_comput
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score


# class FeatureSave(Metric):
#     def __init__(self, num_query, max_rank=50, feat_norm='yes'):
#         super(R1_mAP, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm

#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []
#         self.sceneids = []
#         self.img_path = []

#     def update(self, output):
#         feat, pid, camid, sceneid, img_path = output
#         self.feats.append(feat)
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))
#         self.sceneids.extend(np.asarray(sceneid))
#         self.img_path.extend(img_path)

#     def compute(self):
#         feats = torch.cat(self.feats, dim=0)
#         if self.feat_norm == 'yes':
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)
#         # query


#         return

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.pp_acc = 0.
        self.pv_acc = 0.
        self.fuse_acc = 0.
        self.cnt = 0
        self.score_list = []  # 存储预测得分
        self.score1_list = []  # 存储预测得分
        self.score2_list = []  # 存储预测得分
        self.label_list = []  # 存储真实标签

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.sceneids = []
        self.img_path = []

        self.score1_list = []  # 存储预测得分
        self.score2_list = []  # 存储预测得分
        self.score_list = []  # 存储预测得分
        self.label_list = []  # 存储真实标签

        # print("reset cnt before", self.total_acc)
        self.fuse_acc = 0.
        print("reset total_acc", self.fuse_acc)
        #print("reset cnt before", self.cnt)
        self.cnt = 0
        print("reset cnt ", self.cnt)

    def update(self, output):
        pid, camid, sceneid, img_path, pp_acc, pv_acc, fuse_acc, pred_s_list, fuse_pred_list = output
        # print("feat, pid, camid, sceneid, img_path, batch_acc, pred_list, fuse_pred_list", feat, pid, camid, sceneid, img_path, batch_acc, pred_list, fuse_pred_list)
        # print(batch_acc)
        # print(img_path)
        self.pp_acc += pp_acc.cpu()
        self.pv_acc += pv_acc.cpu()
        self.fuse_acc += fuse_acc.cpu()

        # print("reset total_acc before", self.total_acc)

        self.cnt += 1
        #self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.sceneids.extend(np.asarray(sceneid))
        self.img_path.extend(img_path)
        #print("fuse_pred_list", fuse_pred_list[0].shape)
        self.score_list.extend(np.array(F.softmax(fuse_pred_list, dim=1)))
        #self.score1_list.extend(np.array(F.softmax(pred_s_list[0], dim=1).cpu()))  # .max(1)[1])  # 存储预测得分
        #self.score2_list.extend(np.array(F.softmax(pred_s_list[1], dim=1).cpu()))  # .max(1)[1])  # 存储预测得分
        self.label_list.extend(pid)

    def cal_EER(self, score_list, label_list, path, num_class=840):
        # print("score_list", score_list)
        score_onehot = np.array(score_list)
        # 将label转换成onehot形式
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], num_class)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)
        """
        score_tensor = torch.tensor(score_list)
        score_tensor = score_tensor.reshape((score_tensor.shape[0], 1))
        score_onehot = torch.zeros(score_tensor.shape[0], num_class)
        score_onehot.scatter_(dim=1, index=score_tensor, value=1)
        score_onehot = np.array(score_onehot)
        """
        print("score_array:", score_onehot.shape)  # (batchsize, classnum)
        print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

        # 调用sklearn库，计算每个类别对应的fpr和tpr
        fpr_dict = dict()
        tpr_dict = dict()
        frr_dict = dict()
        roc_auc_dict = dict()

        # micro
        fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_onehot.ravel())
        roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
        frr_dict["micro"] = 1 - tpr_dict["micro"]

        # tb_writer.add_scalars('FV_CBAM',)

        #
        # x = fpr_dict["micro"]
        # y = frr_dict["micro"]
        # np.savetxt('sCA/TJ_x.csv', x, delimiter=',')
        # np.savetxt('sCA/TJ_y.csv', y, delimiter=',')
        # # # #

        # print(frr_dict["micro"] )
        # print(frr_dict["micro"][12])
        # eer["micro"] = np.absolute(frr_dict["micro"] - fpr_dict["micro"])
        # eer = np.nanargmin(eer["micro"])
        # print(eer)
        eer_1 = fpr_dict["micro"][np.nanargmin(np.absolute(frr_dict["micro"] - fpr_dict["micro"]))]
        eer_2 = frr_dict["micro"][np.nanargmin(np.absolute(frr_dict["micro"] - fpr_dict["micro"]))]
        eer = (eer_1 + eer_2) / 2.
        print("eer1:", eer_1)
        print("eer2:", eer_2)
        print(eer)
        f = open("/home/zhangli/CCNet/outputs/" + path + "_err.txt", "a")
        f.write(str(eer) + " ")
        f.flush()
        f.close()
        # macro
        # First aggregate all false positive rates
        # all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
        # # Then interpolate all ROC curves at this points
        # mean_tpr = np.zeros_like(all_fpr)
        # for i in range(num_class):
        #     mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
        # # Finally average it and compute AUC
        # mean_tpr /= num_class
        # fpr_dict["macro"] = all_fpr
        # tpr_dict["macro"] = mean_tpr
        # roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
        # roc_auc[i] = auc(fpr_dict["micro"], tpr_dict["micro"])
        plt.figure()
        lw = 2
        plt.plot(fpr_dict["micro"] * 100, tpr_dict["micro"] * 100, color='darkorange',
                 lw=lw, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig("/home/zhangli/CCNet/err.PNG", )
        plt.close()

        # 绘制所有类别平均的roc曲线
        #    plt.figure()
        #
        #    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
        #             label='ROC curve (area = {0:0.3f})'
        #                   ''.format(roc_auc_dict["micro"]),
        #             color='deeppink', linestyle=':', linewidth=3)
        #
        #    # plt.plot(fpr_dict["micro"], frr_dict["micro"],
        #    #         # label='macro-average ROC curve (area = {0:0.2f})'
        #    #          #      ''.format(roc_auc_dict["macro"]),
        #    #          color='navy', linestyle='-.', linewidth=4)
        #
        #    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        #    # for i, color in zip(range(num_class), colors):
        #    #     plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
        #    #              label='ROC curve of class {0} (area = {1:0.2f})'
        #    #                    ''.format(i, roc_auc_dict[i]))
        #    plt.plot([0, 1], [0, 1], 'k--', lw=1)
        #    plt.xlim([0.0, 1.0])
        #    plt.ylim([0.0, 1.0])
        #    plt.xlabel('False Positive Rate')
        #    plt.ylabel('True Positive Rate')
        #    plt.title('Some extension of Receiver operating characteristic to multi-class')
        #    plt.legend(loc="lower right")
        # #   plt.savefig('set113_roc.jpg')
        #    plt.show()

        #

        # EER
        plt.figure()

        # lw：折线图的线条宽度
        print("fpr", len(fpr_dict["micro"]))
        print("frr", len(frr_dict["micro"]))
        plt.plot(fpr_dict["micro"], frr_dict["micro"],
                 label=' EER = {0:0.3f})'
                       ''.format(eer),
                 color='MediumBlue', linestyle='-', linewidth=2)

        plt.plot([0, 3], [0, 3], 'k--', lw=1)
        plt.xlim([0.0, 0.05])
        plt.ylim([0.0, 0.05])
        plt.xlabel('False Acceptance Rate (%)')
        plt.ylabel('False Rejection Rate (%)')
        #   plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc='lower right')
        #   plt.savefig('set113_roc.jpg')
        plt.savefig("/home/zhangli/CCNet/err.jpg")
        plt.close()
        f = open("/home/zhangli/CCNet/outputs/" + path + "_fpr.txt", "w")
        f.write(str(list(fpr_dict["micro"])) + " ")
        f.flush()
        f.close()
        f = open("/home/zhangli/CCNet/outputs/" + path + "_frr.txt", "w")
        f.write(str(list(frr_dict["micro"])) + " ")
        f.flush()
        f.close()

    def compute(self):
        # feats = torch.cat(self.feats, dim=0)
        # if self.feat_norm == 'yes':
        #    print("The test feature is normalized")
        #    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        # qf = feats[:self.num_query]
        # q_pids = np.asarray(self.pids[:self.num_query])
        # q_camids = np.asarray(self.camids[:self.num_query])

        # q_sceneids = np.asarray(self.sceneids[:self.num_query]) # zxp
        # gallery
        # gf = feats[self.num_query:]
        # g_pids = np.asarray(self.pids[self.num_query:])
        # g_camids = np.asarray(self.camids[self.num_query:])

        # g_sceneids = np.asarray(self.sceneids[self.num_query:]) # zxp

        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)
        print(self.cnt)
        self.pp_acc /= self.cnt
        self.pv_acc /= self.cnt
        self.fuse_acc /= self.cnt

        self.cal_EER(self.score_list, self.label_list, "NUPT1/_cnnhu9_")
        # self.cal_EER(self.score1_list, self.label_list, "_18_no_c1_test")
        # self.cal_EER(self.score2_list, self.label_list, "_18_no_c2_test")
        return self.pp_acc, self.pv_acc, self.fuse_acc


class Param_R1_mAP(Metric):
    def __init__(self, num_query, weight_param, max_rank=50, feat_norm='yes'):
        super(Param_R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        # self.weights = []
        self.weight_parm = weight_param

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        # self.weights = []
        self.sceneids = []
        self.img_path = []

    def update(self, output):
        # feat, weight, pid, camid, sceneid, img_path = output
        feat, pid, camid, sceneid, img_path = output
        self.feats.append(feat)
        # self.weights.append(weight)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.sceneids.extend(np.asarray(sceneid))
        self.img_path.extend(img_path)

    def compute(self):
        l = len(self.feats[0])
        f_list = [[] for _ in range(l)]
        for f in self.feats:
            for i in range(l):
                f_list[i].append(f[i])
        f_list = [torch.cat(f, dim=0) for f in f_list]

        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            f_list = [torch.nn.functional.normalize(f) for f in f_list]

        # query
        qf_list = [feats[:self.num_query] for feats in f_list]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_sceneids = np.asarray(self.sceneids[:self.num_query])  # zxp

        # gallery
        gf_list = [feats[self.num_query:] for feats in f_list]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_sceneids = np.asarray(self.sceneids[self.num_query:])  # zxp

        distmat = param_dist_comput(qf_list, gf_list, self.weight_parm.detach())

        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())

        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)
        # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class Weighted_R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(Weighted_R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.weights = []

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.weights = []
        self.sceneids = []
        self.img_path = []

    def update(self, output):
        feat, weight, pid, camid, sceneid, img_path = output
        self.feats.append(feat)
        self.weights.append(weight)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.sceneids.extend(np.asarray(sceneid))
        self.img_path.extend(img_path)

    def compute(self):
        l = len(self.feats[0])
        f_list = [[] for _ in range(l)]
        w_list = [[] for _ in range(l)]
        for f, w in zip(self.feats, self.weights):
            for i in range(l):
                f_list[i].append(f[i])
                w_list[i].append(w[i])
        f_list = [torch.cat(f, dim=0) for f in f_list]
        w_list = [torch.cat(w, dim=0) for w in w_list]

        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            f_list = [torch.nn.functional.normalize(f) for f in f_list]

        # query
        qf_list = [feats[:self.num_query] for feats in f_list]
        qw_list = [w[:self.num_query] for w in w_list]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        q_sceneids = np.asarray(self.sceneids[:self.num_query])  # zxp
        # gallery
        gf_list = [feats[self.num_query:] for feats in f_list]
        gw_list = [w[self.num_query:] for w in w_list]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        g_sceneids = np.asarray(self.sceneids[self.num_query:])  # zxp

        distmat = weighted_dist_comp(qf_list, gf_list, qw_list, gw_list)

        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())

        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)
        # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class R1_mAP_reranking(Metric):

    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.sceneids = []
        self.img_path = []

    def update(self, output):
        feat, pid, camid, sceneid, img_path = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.sceneids.extend(np.asarray(sceneid))
        self.img_path.extend(img_path)

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        q_sceneids = np.asarray(self.sceneids[:self.num_query])  # zxp
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        g_sceneids = np.asarray(self.sceneids[self.num_query:])  # zxp

        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)

        return cmc, mAP

    # def __init__(self, num_query, max_rank=50, feat_norm='yes'):
    #     super(R1_mAP_reranking, self).__init__()
    #     self.num_query = num_query
    #     self.max_rank = max_rank
    #     self.feat_norm = feat_norm

    # def reset(self):
    #     self.feats = []
    #     self.pids = []
    #     self.camids = []

    # def update(self, output):
    #     feat, pid, camid = output
    #     self.feats.append(feat)
    #     self.pids.extend(np.asarray(pid))
    #     self.camids.extend(np.asarray(camid))

    # def compute(self):
    #     feats = torch.cat(self.feats, dim=0)
    #     if self.feat_norm == 'yes':
    #         print("The test feature is normalized")
    #         feats = torch.nn.functional.normalize(feats, dim=1, p=2)

    #     # query
    #     qf = feats[:self.num_query]
    #     q_pids = np.asarray(self.pids[:self.num_query])
    #     q_camids = np.asarray(self.camids[:self.num_query])
    #     # gallery
    #     gf = feats[self.num_query:]
    #     g_pids = np.asarray(self.pids[self.num_query:])
    #     g_camids = np.asarray(self.camids[self.num_query:])
    #     # m, n = qf.shape[0], gf.shape[0]
    #     # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #     #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    #     # distmat.addmm_(1, -2, qf, gf.t())
    #     # distmat = distmat.cpu().numpy()
    #     print("Enter reranking")
    #     distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
    #     #pdb.set_trace()
    #     cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)    # modified

    #     return cmc, mAP