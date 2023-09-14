import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
from AENet import ae


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 3.5).unsqueeze(1)
    _2D_window =_1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window.cuda()


# Check CUDA's presence
cuda_is_present = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda_is_present else torch.FloatTensor

def to_cuda(data):
    	return data.cuda() if cuda_is_present else data



class STD(torch.nn.Module):
    def __init__(self, window_size = 5):
        super(STD, self).__init__()
        self.window_size = window_size
        self.channel=1
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.window=create_window(self.window_size, self.channel)
        self.window.to(torch.device('cuda'))
    def forward(self, img):
        mu = F.conv2d(img, self.window, padding = self.window_size//2, groups = self.channel)
        mu_sq = mu.pow(2)
        sigma_sq = F.conv2d(img*img, self.window, padding = self.window_size//2, groups = self.channel)  - mu_sq
        B,C,W,H=sigma_sq.shape
        sigma_sq=torch.flatten(sigma_sq, start_dim=1)
        noise_map = self.softmax(sigma_sq)
        noise_map=torch.reshape(noise_map,[B,C,W,H])
        return noise_map


class NCMSE(nn.Module):
    def __init__(self):
        super(NCMSE, self).__init__()
        self.std=STD()
    def forward(self, out_image, gt_image, org_image):
        loss = torch.mean(torch.pow(out_image - org_image, 2))
        return loss




class SNDisLoss(torch.nn.Module):
    """
    The loss for sngan discriminator
    """
    def __init__(self,weight = 0.5):
        super(SNDisLoss, self).__init__()
        self.weight = weight    

    def forward(self, pos, neg, out_image, org_image):
        #return self.weight*(torch.mean(torch.pow(pos - neg, 2)) + self.weight*torch.mean(torch.pow(out_image - org_image, 2)))
        return -torch.mean(pos) + torch.mean(neg)
    
    
    



class SNGenLoss(torch.nn.Module):
    """
    The loss for sngan generator
    """
    def __init__(self, weight = 0.5):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def forward(self, out_image,gt_image):
        loss = ((self.weight * (out_image - gt_image)) ** 2).mean()
        return loss




class p_loss(torch.nn.Module):
    """
    Perceptual loss
    """
    def __init__(self):
        super(p_loss, self).__init__()
        self.L2 = nn.MSELoss()
        
    def forward(self, pred, gt_image, save_path):
         
        whole_model = torch.load('./save_pretrain_ae/' + 'latest_ckpt_pretrain.pth.tar') 
        netG_state_dict = whole_model['netG_state_dict']        
        
        netG = ae()
        netG = netG.cuda() 
        netG.load_state_dict(netG_state_dict)
                
        for x in netG.parameters():  
            x.requires_grad = False     

        pred_feature1, pred_feature2, pred_feature3, pred_feature4, pred_feature5, pred_feature6, pred_feature7, pred_feature8 = netG(pred)
        gt_feature1, gt_feature2, gt_feature3, gt_feature4, gt_feature5, gt_feature6, gt_feature7, gt_feature8 = netG(gt_image)
        
        loss1 = self.L2(pred_feature1, gt_feature1)
        loss2 = self.L2(pred_feature2, gt_feature2)
        loss4 = self.L2(pred_feature4, gt_feature4)
   
        
        loss = loss1 + loss2 + loss4
        return loss
