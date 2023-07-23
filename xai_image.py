import os
import sys
import gc

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

def main(cfg):

    # 更新する画像の枚数
    xai_img_num = 10

    # 対象クラスのインデックス
    cls_index = 0
    
    # ----------------------------
    # Model
    # ----------------------------
    if cfg["model_load_flag"] in [1]:
        model_dict = torch.load(cfg["in_model_path"])
        #print(torch.load(cfg["in_model_path"])['emb_word.weight'].shape)
    else:
        model_dict = None

    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    img_cls_num = 1000

    model = XAI_image(
        img_model = resnet,
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
    train_y = [cls_index for _ in range(xai_img_num)]
    train_y = np.array(train_y)
    train_y = torch.tensor(train_y, dtype=torch.int64)
    train_y.to('cuda')

    #print(model.emb_img(train_x).cpu().detach().item())
    #exit()
    
    for epoch in tqdm(range(cfg["epoch_num"])):
        start = time.time()
        with open("log.csv","a") as f:
            print("Epoch >>>>> ", epoch + epoch_offset)
            
            model.train()
            train_loss = 0.
            train_accuracy = 0
            train_total = 0
    
            opt.zero_grad()

            train_pred_cnt = [0 for i in range(img_cls_num)]
            each_label_accuracy = [0 for i in range(img_cls_num)]
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            output_train_x, emb_img = model(train_x)
            output_train_x.to('cuda')
            loss = nn.functional.nll_loss(output_train_x, train_y)

            for i in range(xai_img_num):
                np_img = np.zeros((model.matrix_size,model.matrix_size,model.channel_size), np.float32)
                np_emb_img = emb_img[i]
                for c in range(model.channel_size):
                    for x in range(model.matrix_size):
                        for y in range(model.matrix_size):
                            ii = c*model.matrix_size*model.matrix_size + x*model.matrix_size + y
                            np_img[y,x,c] = np_emb_img[ii]
                print(np_img)
            
            loss.backward()
            opt.step()
                    
            train_loss += loss.item()

            _, predicted = torch.max(output_train_x, 1)
            for i,true_label_tensor in enumerate(train_y):
                true_label = true_label_tensor.cpu().detach().item()
                pred_label = predicted[i].cpu().detach().item()
                each_label_accuracy[true_label] += (true_label == pred_label)
                train_pred_cnt[pred_label] += 1
            train_accuracy += each_label_accuracy[cls_index]
            train_total += train_y.size(0)
            
            if np.isnan(train_loss):
                print("nan")
                exit()
            
            ###
            # GPUの解放
            ###
            del loss
            gc.collect()
            torch.cuda.empty_cache() # <-
                    
            #print("mode1_each", each_label_accuracy, np.sum(np.array(each_label_accuracy)) )
    
            t_loss = (train_loss / train_total)
            t_accuracy = (100 * train_accuracy / train_total)
            print("train (loss, accuracy)",t_loss,t_accuracy)
            writer.add_scalar("loss/train", t_loss, epoch + epoch_offset)
            writer.add_scalar("accuracy/train", t_accuracy, epoch + epoch_offset)

            if t_loss < best_train_loss:
                best_train_loss = t_loss
                # ----------------------------
                # save model
                # ----------------------------
                print(f"model saved as model_best_train.pth")
                torch.save(model.state_dict(), "model_best_train.pth")
            torch.save(model.state_dict(), "model_latest.pth")

            elapsed_time = time.time() - start
            f.write(f"{epoch + epoch_offset},{t_loss},{t_accuracy},{elapsed_time}\n")
    writer.close()



if __name__ == "__main__":

    with open(os.path.join('conf','parameters.yaml'),'r') as yml:
        cfg = yaml.safe_load(yml)
        model, train_loader, pre = main(cfg)
