import os
import sys
import gc
import cv2

import numpy as np
import random
from tqdm import tqdm
import pickle
import yaml

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from model import *
import time
import pprint

from multiprocessing import Value, Array, Process, Manager 
from PIL import Image
from torchvision import transforms
from torchvision import models

def main(cfg):

    # 更新する画像の枚数
    xai_img_num = 1

    # 対象クラスのインデックス
    target_cls = 567
    true_cls = None
    # ----------------------------
    # Model
    # ----------------------------
    if cfg["model_load_flag"] in [1]:
        model_dict = torch.load(cfg["in_model_path"])
        #print(torch.load(cfg["in_model_path"])['emb_word.weight'].shape)
    else:
        model_dict = None

    #resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    #resnet = models.resnet50(pretrained=True)
    vitb16 = models.vit_b_16(pretrained=True)
    #exit()
    
    img_cls_num = 1000
    
    #base_img_path = ""
    #base_img_path = "2.jpg"
    base_img_path = "0_17.png"
    #base_img_path = ""
    
    model = XAI_image(
        base_img_path = base_img_path,
        img_model = vitb16,
        cls_num = img_cls_num,
        img_num = xai_img_num, 
        batch_size = cfg["batch_size"], 
        init_mode = cfg["init_mode"], 
        model_load_mode = cfg["model_load_flag"],
        model_dict = model_dict
    )
    
    model.to("cuda")
    #model = torch.nn.DataParallel(model)
    # ----------------------------
    # Optimizer, loss
    # ----------------------------
    optimizer = optim.RAdam
    opt = optimizer(model.parameters(), lr = cfg["learning_rate"])

    # log_dirでlogのディレクトリを指定
    writer = SummaryWriter(log_dir="./logs")
    epoch_offset = cfg["epoch_offset"]
    
    # ----------------------------
    # Training loop
    # ----------------------------
    loss_per_epch = []
    matrix_mean = []
    matrix_mean_epoch = []

    last_i_mat = None
    best_train_loss = 10000000000
    best_validation_loss = 10000000000
    best_validation_accuracy = 0
    nan_flag = False

    train_x = [i for i in range(xai_img_num)]
    train_x = np.array(train_x)
    train_x = torch.tensor(train_x, dtype=torch.int64)
    train_x.to('cuda')
    train_y = [target_cls for _ in range(xai_img_num)]
    train_y = np.array(train_y)
    train_y = torch.tensor(train_y, dtype=torch.int64)
    train_y.to('cuda')

    #print(model.emb_img(train_x).cpu().detach().item())
    #exit()
    
    model.eval()

    if model.norm_flag:
        output_check = model.check(model.base_img)
        _, predicted = torch.max(output_check, 1)
        pred_label = predicted[0].cpu().detach().item()
        print(pred_label, F.softmax(output_check[0],dim=0)[pred_label])
        true_cls = pred_label
        #print(pred_label, output_check[0][pred_label])
        #exit()
    
    for epoch in tqdm(range(cfg["epoch_num"])):
        start = time.time()
        print("Epoch >>>>> ", epoch + epoch_offset)
        
        #model.train()
        train_loss = 0.
        train_accuracy = 0
        train_total = 0

        opt.zero_grad()

        train_pred_cnt = [0 for i in range(img_cls_num)]
        each_label_accuracy = [0 for i in range(img_cls_num)]
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        output_train_x, logsoftmax, emb_img = model(train_x)
        output_train_x.to('cuda')
        logsoftmax.to('cuda')
        loss = nn.functional.nll_loss(logsoftmax, train_y)

        for i in range(xai_img_num):
            np_img = np.zeros((model.matrix_size,model.matrix_size,model.channel_size), np.float64)
            np_emb_img = emb_img[i]
            for c in range(model.channel_size):
                for x in range(model.matrix_size):
                    for y in range(model.matrix_size):
                        #ii = c*model.matrix_size*model.matrix_size + x*model.matrix_size + y
                        np_img[y,x,c] = np_emb_img[c,y,x]
            np_img = 255*np_img
            print(np.min(np_img),np.max(np_img))
            np_img = np_img.astype(np.uint8)
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            save_path = f"{i}_{epoch}.png"
            cv2.imwrite(save_path, np_img)
            print(f"saved as {save_path}")
            
        loss.backward()
        
        emb_grad = model.emb_img.weight.grad
        emb_grad_absmean = torch.mean(torch.absolute(emb_grad))
        #update_coef = 1./5#1./255.
        update_coef = 1./255.
        print("emb_grad_absmean",emb_grad_absmean)
        p_emb_grad_thresh = torch.nn.Threshold(emb_grad_absmean,0.)(emb_grad)
        p_emb_grad_sign = torch.sign(p_emb_grad_thresh)
        n_emb_grad_thresh = torch.nn.Threshold(emb_grad_absmean,0.)(-emb_grad)
        n_emb_grad_sign = torch.sign(n_emb_grad_thresh)
        for i in range(xai_img_num):
            for j in range(model.channel_size*model.matrix_size**2):
                model.emb_img.weight.data[i][j] = model.emb_img.weight.data[i][j] - update_coef*(p_emb_grad_sign[i][j] - n_emb_grad_sign[i][j])

        #opt.step()
                
        train_loss += loss.item()
        
        model.eval()
        
        _, predicted = torch.max(logsoftmax, 1)
        for i,true_label_tensor in enumerate(train_y):
            true_label = true_label_tensor.cpu().detach().item()
            pred_label = predicted[i].cpu().detach().item()
        train_total += train_y.size(0)
        
        if np.isnan(train_loss):
            print("nan")
            exit()

        if model.norm_flag:
            print(true_cls, target_cls, pred_label, F.softmax(output_train_x)[0][true_cls], F.softmax(output_train_x)[0][target_cls])
        else:
            print(target_cls, pred_label, F.softmax(output_train_x)[0][target_cls])
            
        ###
        # GPUの解放
        ###
        del loss
        gc.collect()
        torch.cuda.empty_cache() # <-
                
        #print("mode1_each", each_label_accuracy, np.sum(np.array(each_label_accuracy)) )

        t_loss = (train_loss / train_total)
        print("train (loss)",t_loss)

        if t_loss < best_train_loss:
            best_train_loss = t_loss
            # ----------------------------
            # save model
            # ----------------------------
            print(f"model saved as model_best_train.pth")
            torch.save(model.state_dict(), "model_best_train.pth")
        torch.save(model.state_dict(), "model_latest.pth")

        elapsed_time = time.time() - start
    writer.close()



if __name__ == "__main__":

    with open(os.path.join('conf','parameters.yaml'),'r') as yml:
        cfg = yaml.safe_load(yml)
        main(cfg)
