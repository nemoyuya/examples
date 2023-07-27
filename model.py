import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np

from PIL import Image
from torchvision import transforms
import cv2

class XAI_image(nn.Module):
    
    def __init__(self, base_img_path, img_model, cls_num, img_num, batch_size, init_mode, model_load_mode, model_dict=None):
        super().__init__()
        self.norm_flag = base_img_path != ""
        self.mean_1 = 0.485
        self.mean_2 = 0.456
        self.mean_3 = 0.406
        self.std_1 = 0.229
        self.std_2 = 0.224
        self.std_3 = 0.225
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean_1,self.mean_2,self.mean_3], std=[self.std_1,self.std_2,self.std_3]),
        ])
        self.img_model = img_model.cuda()
        self.img_model.eval()
        self.batch_size = batch_size
        self.img_num = img_num
        self.matrix_size = 224
        self.channel_size = 3
        self.emb_size = self.channel_size*self.matrix_size**2
        self.emb_img = nn.Embedding(img_num, self.emb_size)
        
        # ---------------------------
        # Initialize the internale weight 
        # ---------------------------
        if init_mode == "uniform":
            self.emb_img.weight.data.uniform_(-1, 1)
        elif init_mode == "gauss":
            self.emb_img.weight.data.normal_(0, 0.1)            
        else :
            raise NotImplementedError(init_mode)

        
        if self.norm_flag:
            #self.eval()
            input_image = Image.open(base_img_path)
            self.base_img = transforms.ToTensor()(input_image)
            self.base_img = transforms.Normalize(mean=[self.mean_1,self.mean_2,self.mean_3], std=[self.std_1,self.std_2,self.std_3])(self.base_img)
            #self.base_img = self.preprocess(input_image)
            self.base_img = self.base_img.unsqueeze(0)
            self.base_img = self.base_img.to('cuda')
            output_check = self.check(self.base_img)
            _, predicted = torch.max(output_check, 1)
            pred_label = predicted[0].cpu().detach().item()
            print(pred_label, F.softmax(output_check[0],dim=0)[pred_label])

            tmp_img = transforms.Normalize(mean=[-self.mean_1/self.std_1, -self.mean_2/self.std_2, -self.mean_3/self.std_3], std=[1./self.std_1, 1./self.std_2, 1./self.std_3])(self.base_img)
            #tmp_img = self.base_img
            np_tmp_img = tmp_img.to('cpu').detach().numpy().copy()
            np_tmp_img = np_tmp_img[0]
            np_tmp_img = np_tmp_img.transpose(1,2,0)
            np_tmp_img *= 255.
            np_tmp_img = np_tmp_img.astype(np.uint8)
            np_tmp_img = cv2.cvtColor(np_tmp_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"np_tmp_img.png",np_tmp_img)

            input_image = Image.open("np_tmp_img.png")
            test_img = transforms.ToTensor()(input_image)
            test_img = transforms.Normalize(mean=[self.mean_1,self.mean_2,self.mean_3], std=[self.std_1,self.std_2,self.std_3])(test_img)
            #test_img = self.preprocess(input_image)
            test_img = test_img.unsqueeze(0)
            test_img = test_img.to('cuda')
            #test_img = test_img.cuda()
            check = self.check(test_img)
            check.to('cuda')
            _, predicted = torch.max(check, 1)
            pred_label = predicted[0].cpu().detach().item()
            print(pred_label, F.softmax(check[0],dim=0)[pred_label])
            
            for c in range(self.channel_size):
                for x in range(self.matrix_size):
                    for y in range(self.matrix_size):
                        i = c*self.matrix_size*self.matrix_size + y*self.matrix_size + x
                        self.emb_img.weight.data[0][i] = torch.logit(torch.tensor(tmp_img[0,c,y,x]))
        else:
            self.base_img = np.zeros((self.matrix_size,self.matrix_size,3),np.float32)
            self.base_img = transforms.ToTensor()(self.base_img)
            self.base_img = self.base_img.unsqueeze(0)
            self.base_img = self.base_img.to('cuda')
                       
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
        emb = torch.sigmoid(emb)
        emb = emb.reshape(self.img_num, self.channel_size, self.matrix_size, self.matrix_size)
        if self.norm_flag:
            norm_emb = transforms.Normalize(mean=[self.mean_1, self.mean_2, self.mean_3], std=[self.std_1, self.std_2, self.std_3])(emb)
            x = self.img_model(norm_emb)
        else:
            x = self.img_model(emb)
        logsoftmax = F.log_softmax(x, dim=1)
        
        return x, logsoftmax, emb

    
    def check(self, x):
        
        x = self.img_model(x)
        #x = F.softmax(x, dim=1)
        #x = F.log_softmax(x, dim=1)
        
        return x

    
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
