#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Linyao Gao
@Contact: linyaog@sjtu.edu.cn
@File: test.py
@Time: 2021/01/04
"""
import collections

import open3d
import os
import numpy as np
import torch
from dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.pc_error_wrapper import pc_error
from iostream import IOStream
import time
import importlib
import sys
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# 定义测试指标

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('point_based_PCGC')
    parser.add_argument('--dataset_path', type=str, default='../pointcloud_compression/PointCloudDatasets')
    return parser.parse_args()

def cal_bpp(likelihood, device, num_points, batch_size):
    bpp = torch.sum(torch.log(likelihood)) / -(torch.log(torch.Tensor([2.0]).to(device))) / (float(num_points)*float(batch_size))
    return bpp
    
def cal_d1(pc_gt, decoder_output, step, checkpoint_path):
    # 原始点云写入ply文件
    ori_pcd = open3d.geometry.PointCloud() # 定义点云
    ori_pcd.points = open3d.utility.Vector3dVector(np.squeeze(pc_gt)) # 定义点云坐标位置[N,3]
    orifile = checkpoint_path+'/pc_file/'+'d1_ori_'+str(step)+'.ply'# 保存路径
    open3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)
    # 重建点云写入ply文件
    rec_pcd = open3d.geometry.PointCloud()
    rec_pcd.points = open3d.utility.Vector3dVector(np.squeeze(decoder_output))
    recfile = checkpoint_path+'/pc_file/'+'d1_rec_'+str(step)+'.ply'
    open3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)
    
    pc_error_metrics = pc_error(infile1=orifile, infile2=recfile, res=2) # res为数据峰谷差值
    pc_errors = [pc_error_metrics["mse1,PSNR (p2point)"][0], 
                pc_error_metrics["mse2,PSNR (p2point)"][0], 
                pc_error_metrics["mseF,PSNR (p2point)"][0],
                pc_error_metrics["mse1      (p2point)"][0],
                pc_error_metrics["mse2      (p2point)"][0],
                pc_error_metrics["mseF      (p2point)"][0]]
    return pc_errors

def cal_d2(pc_gt, decoder_output, step, checkpoint_path):

    # 原始点云写入ply文件
    ori_pcd = open3d.geometry.PointCloud() # 定义点云
    ori_pcd.points = open3d.utility.Vector3dVector(np.squeeze(pc_gt)) # 定义点云坐标位置[N,3]
    ori_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30)) # 计算normal
    orifile = checkpoint_path+'/pc_file/'+'d2_ori_'+str(step)+'.ply'# 保存路径
    print(orifile)
    open3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)
    # 将ply文件中normal类型double转为float32
    lines = open(orifile).readlines()
    to_be_modified = [7, 8, 9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double','float32')
    file = open(orifile, 'w')
    for line in lines:
        file.write(line)
    file.close()
    # 可视化点云,only xyz
    # open3d.visualization.draw_geometries([ori_pcd])
    

    # 重建点云写入ply文件
    rec_pcd = open3d.geometry.PointCloud()
    rec_pcd.points = open3d.utility.Vector3dVector(np.squeeze(decoder_output))
    # rec_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30)) # 计算normal
    recfile = checkpoint_path+'/pc_file/'+'d2_rec_'+str(step)+'.ply'
    open3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

    pc_error_metrics = pc_error(infile1=orifile, infile2=recfile, normal=True, res=2) # res为数据峰谷差值,normal=True为d2 
    pc_errors = [pc_error_metrics["mse1,PSNR (p2plane)"][0], 
                pc_error_metrics["mse2,PSNR (p2plane)"][0], 
                pc_error_metrics["mseF,PSNR (p2plane)"][0],
                pc_error_metrics["mse1      (p2plane)"][0],
                pc_error_metrics["mse1      (p2plane)"][0],
                pc_error_metrics["mse1      (p2plane)"][0],
                pc_error_metrics["mse2      (p2plane)"][0],
                pc_error_metrics["mseF      (p2plane)"][0]]

    return pc_errors

def test(model, args, batch_size=1):
    
    checkpoint_path = '.'

    test_data = Dataset(root=args.dataset_path, dataset_name='shapenetcorev2', num_points=2048, split='test')

    test_loader = DataLoader(test_data, num_workers=2, batch_size=batch_size, shuffle=False)


    # 初始化变量
    avg_chamfer_dist = np.array([0.0 for i in range(55)])
    avg_d1_psnr = np.array([0.0 for i in range(55)])
    avg_d1_mse = np.array([0.0 for i in range(55)])
    avg_d2_psnr = np.array([0.0 for i in range(55)])
    avg_d2_mse = np.array([0.0 for i in range(55)])
    avg_bpp = np.array([0.0 for i in range(55)])
    counter = np.array([0.0 for i in range(55)])
    total_chamfer_dist = 0.0
    total_d1_psnr = 0.0
    total_d1_mse = 0.0
    total_d2_psnr = 0.0
    total_d2_mse = 0.0
    total_bpp = 0.0

    num_samples = 0

    for step, data in enumerate(test_loader):
        with torch.no_grad():
            pc_data = data[0]
            label = data[1]

            if torch.cuda.is_available():
                pc_gt = pc_data.cuda()
                pc_data = pc_data.cuda().transpose(1,2) # [B,N,3]

            bpp, decoder_output, _ = model(pc_data)

            avg_bpp[label] += bpp
            total_bpp += bpp

            # 转换成numpy
            pc_gt = pc_gt.cpu().detach().numpy()
            decoder_output = decoder_output.cpu().detach().numpy()
            # D1 psnr & D1 mse
            d1_results = cal_d1(pc_gt, decoder_output, step, checkpoint_path)
            d1_psnr = d1_results[2].item()
            d1_mse = d1_results[5].item()
            avg_d1_mse[label] += d1_mse
            total_d1_mse += d1_mse
            avg_d1_psnr[label] += d1_psnr
            total_d1_psnr += d1_psnr
            # D2 psnr & D2 mse
            d2_results = cal_d2(pc_gt, decoder_output, step, checkpoint_path)
            d2_psnr = d2_results[2].item()
            d2_mse = d2_results[5].item()
            avg_d2_mse[label] += d2_mse
            total_d2_mse += d2_mse
            avg_d2_psnr[label] += d2_psnr
            total_d2_psnr += d2_psnr
            # 打印
            # print("step:", step, "bpp:", bpp)
            # print("step:", step, "d1_psnr:", d1_psnr)
            # print("step:", step, "d1_mse:", d1_mse)
            # print("step:", step, "chamfer_dist:", chamfer_dist)
            # print("step:", step, "d2_psnr:", d2_psnr)
            # print("step:", step, "d2_mse:", d2_mse)

        counter[label] += 1
        num_samples += 1
    for i in range(55):
        avg_chamfer_dist[i] /= counter[i]
        avg_d1_psnr[i] /= counter[i]
        avg_d1_mse[i] /= counter[i]
        avg_d2_psnr[i] /= counter[i]
        avg_d2_mse[i] /= counter[i]
        avg_bpp[i] /= counter[i]
    total_chamfer_dist /= num_samples
    total_d1_psnr /= num_samples
    total_d2_mse /= num_samples
    total_d2_psnr /= num_samples
    total_d2_mse /= num_samples
    total_bpp /= num_samples

    # print("Average_Chamfer_Dist:", avg_chamfer_dist)
    # print("Average_D1_PSNR:", avg_d1_psnr)
    # print("Average_D1_mse:", avg_d1_mse)
    # print("Average_D2_PSNR:", avg_d2_psnr)
    # print("Average_D2_mse:", avg_d2_mse)
    # print("Average_bpp:", avg_bpp)

    for i in range(55):
        outstr = str(i)+" Average_bpp: %.6f, Average_D1_PSNR: %.6f, Average_D1_mse: %.6f, Average_Chamfer_Dist: %.6f, Average_D2_PSNR: %.6f, Average_D2_mse: %.6f\n" % (avg_bpp[i],
            avg_d1_psnr[i], avg_d1_mse[i], avg_chamfer_dist[i], avg_d2_psnr[i], avg_d2_mse[i])
        print(outstr)
    outstr = "Average_bpp: %.6f, Average_D1_PSNR: %.6f, Average_D1_mse: %.6f, Average_Chamfer_Dist: %.6f, Average_D2_PSNR: %.6f, Average_D2_mse: %.6f\n" % (
             total_bpp,
             total_d1_psnr, total_d1_mse, total_chamfer_dist, total_d2_psnr, total_d2_mse)
    print(outstr)



if __name__ == '__main__':

    args = parse_args()
    ckpt_of_different_rates = ['softmaxfinal14000_256_2400','softmaxfinal2000_256_2400','softmaxfinal11000_256_2400','softmaxfinal1500_256_2400','softmaxfinal1100_256_2400','softmaxfinal110_256_2400']
    for exp_name in ckpt_of_different_rates:
        model_name = 'NGS_PCC'
        experiment_dir = 'log/'+model_name+'/'+exp_name
        MODEL = importlib.import_module(model_name)
        model = MODEL.get_model(use_hyperprior=True, bottleneck_size=256, recon_points=2400).cuda()
        model.eval()
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        start_time = time.time()
        test(model, args)
        end_time = time.time()
        outstr = "test_time: %.6f" % ((end_time-start_time)//3600.0)
        print(outstr)




