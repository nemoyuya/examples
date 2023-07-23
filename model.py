import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch.optim as optim

import numpy as np

class XAI_image(nn.Module):
    
    def __init__(self, img_model, cls_num, img_num, batch_size, init_mode, model_load_mode, model_dict=None):
        super().__init__()

        self.img_model = img_model
        self.batch_size = batch_size
        self.matrix_size = 64
        self.channel_size = 3
        self.img_num = img_num
        self.emb_size = self.channel_size*self.matrix_size**2

        self.emb_img = nn.Embedding(img_num, self.emb_size)

        self.semantics_dim = 200

        self.fc1_for_mode1 = nn.Linear(self.emb_size, self.emb_size, bias = False)
        self.fcout_mode1 = nn.Linear(self.emb_size, cls_num, bias = False)
        
        # ---------------------------
        # Initialize the internale weight 
        # ---------------------------
        if init_mode == "uniform":
            self.emb_img.weight.data.uniform_(-1, 1)

        elif init_mode == "gauss":
            self.emb_img.weight.data.normal_(0, 0.1)
            
        else :
            raise NotImplementedError(init_mode)

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        if model_load_mode == 1:
            self.load_state_dict(model_dict)

        #self.emb_img.weight.requires_grad = False

        self.img_model.requires_grad = False
        #print(self.img_model)
        #exit()
        
        return

    
    def forward(self, x):
        
        emb = self.emb_img(x)
        x = emb.reshape(self.img_num, self.channel_size, self.matrix_size, self.matrix_size)
        x = self.img_model(x)
        x = F.log_softmax(x, dim=1)
        
        return x, emb

    
    #def forward(self, x):
    #    
    #    matrix_input = self.emb_img(x)
    #    x = torch.flatten(matrix_input, start_dim=1)
    #    x = F.gelu(self.fc1_for_mode1(x))
    #    x = self.fcout_mode1(x)
    #    x = F.log_softmax(x, dim=1)
    #    #x = torch.sigmoid(x)
    #    
    #    return x
