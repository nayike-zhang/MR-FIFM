# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import pdb
import os
import os.path as osp
import numpy as np
from .bases import BaseImageDataset


class VERI_MM(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'image_gan'

    def __init__(self, root='/home/haoluo/data', verbose=False, **kwargs):
        super(VERI_MM, self).__init__()
        root = r"F:\datasets\VeRi776_multimodal"
        self.dataset_dir = "/home/zhangli/CCNet/AAAI2022-IEEE-for-MMReID-main/NUPT"##osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train/PP_ROI_tr')
        self.query_dir = osp.join(self.dataset_dir, 'test/PP_ROI_tt')
        self.gallery_dir = osp.join(self.dataset_dir, 'test/PV_ROI_tt')

        self._check_before_run()

        train = self._process_train_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=True)
        #print("train", train)
        #print("query", query)
        gallery = self._process_dir(self.gallery_dir, relabel=True)
        #pdb.set_trace()
        if verbose:
            print("=> RGB_IR loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)  # (300, 900, 1)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        print("veri776 multimodal version used as dataset! => {}".format(self.dataset_dir))
        #pdb.set_trace()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_train_dir(self, dir_path, relabel=False):
        file_paths = glob.glob(dir_path + "/*")
        path_list = os.listdir(dir_path)
        path_array = np.array(path_list).astype(np.int16)
        #print("path_list",path_array)
        #print("file_paths",file_paths)
        path_array.sort()
        #print("path_list",path_array)
        pid_container = set()
        data = []
        cnt = 0
        #print("file_paths",file_paths)
        for file_path in path_array:
            img_paths_RGB = glob.glob(osp.join(dir_path, str(file_path), '*.jpg'))

            for img_path_RGB in img_paths_RGB:
                jpg_name = img_path_RGB.split('/')[-1]
                pid = file_path
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            for img_path_RGB in img_paths_RGB:
                img = []
                jpg_name = img_path_RGB.split('/')[-1]
                #print("jpg_name",jpg_name)
                img_path_NI = osp.join("/home/zhangli/CCNet/AAAI2022-IEEE-for-MMReID-main/NUPT/train/PV_ROI_tr/", str(file_path), jpg_name)

                #img_path_TI = osp.join(dir_path, 'TI', jpg_name)
                img.append(img_path_RGB)
                img.append(img_path_NI)
                #print("img",img)
                #img.append(img_path_TI)
                pid = file_path
                camid = cnt
                cnt+=1 #int(jpg_name.split('_')[1][3])
                #camid -= 1 # index starts from 0
                if relabel:
                    pid = pid2label[pid]
                data.append(((img_path_RGB, img_path_NI, img_path_NI), pid, 0, camid))
        return data
    def _process_dir(self, dir_path, relabel=False):
        file_paths = glob.glob(dir_path + "/*")
        path_list = os.listdir(dir_path)
        path_array = np.array(path_list).astype(np.int16)

        #print("path_list",path_array)
        #print("file_paths",file_paths)
        path_array.sort()
        #print("path_list",path_array)
        pid_container = set()
        data = []
        cnt = 0
        #print("file_paths",file_paths)
        for file_path in path_array:
            img_paths_RGB = glob.glob(osp.join(dir_path, str(file_path), '*.jpg'))
            # print("img_paths_RGB",img_paths_RGB)
            for img_path_RGB in img_paths_RGB:
                jpg_name = img_path_RGB.split('/')[-1]
                pid = file_path
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            for img_path_RGB in img_paths_RGB:
                img = []
                jpg_name = img_path_RGB.split('/')[-1]
                #print("jpg_name",jpg_name)
                img_path_NI = osp.join("/home/zhangli/CCNet/AAAI2022-IEEE-for-MMReID-main/NUPT/test/PV_ROI_tt/", str(file_path), jpg_name)

                #img_path_TI = osp.join(dir_path, 'TI', jpg_name)
                img.append(img_path_RGB)
                img.append(img_path_NI)
                #print("img",img)
                #img.append(img_path_TI)
                pid = file_path
                camid = cnt
                cnt+=1 #int(jpg_name.split('_')[1][3])
                #camid -= 1 # index starts from 0
                if relabel:
                    pid = pid2label[pid]
                data.append([[img_path_RGB, img_path_NI, img_path_NI], pid, 0, camid])
        return data
