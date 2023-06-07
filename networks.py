import numpy as np
import torch
import torch.nn as nn


class Sa_GC(nn.Module):
    def __init__(self,
                 inplanes = 64,
                 ratio = 0.5,
                 pooling_type='att',
                 fusion_types=('channel_add')):
        super(Sa_GC, self).__init__()
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.LogSoftmax(dim=1)

       
        
        self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out
        
        
        
class AGCBlock(nn.Module):
    def __init__(self,image_size = 64,k = 15,stride = 7,padding = 0, inplanes=64):
        super(AGCBlock, self).__init__()
        self.stride = stride
        self.k = k
        self.padding = padding
        self.inplanes = inplanes
        self.unfold = torch.nn.Unfold(self.k, 1, self.padding, self.stride) # [N,c*k*k,l]
        self.fold = torch.nn.Fold(output_size=(image_size, image_size), kernel_size=(self.k,self.k),
                                  dilation=1, padding=0, stride=self.stride) # [N,c*k*k,l]
        self.GC = nn.Sequential(Sa_GC(inplanes=self.inplanes, ratio=0.5, pooling_type='att', fusion_types=('channel_add'))) 

    def forward(self, x): # x: [N,C,H,W] 
        # [N, C, 1, 1]
        batch, channel, height, width = x.shape
        
        
        x_split = x
        x_split_unfold = self.unfold(x_split).transpose(2,1).contiguous() #[N,l,c*k*k]
        batch_unfold, number_unfold, patchsize_unfold = x_split_unfold.size()
        x_split_unfold =  x_split_unfold.view(-1, channel, self.k, self.k) #[N*l,c,k,k]
        x_split_unfold_gc = self.GC(x_split_unfold) #[N*l,c,k,k]
        x_split_unfold_gc = x_split_unfold_gc.view(batch_unfold, number_unfold, patchsize_unfold) #[N,l,c*k*k]
        x_split_unfold_gc = x_split_unfold_gc.transpose(2,1).contiguous() # [N,c*k*k,l]
        x_split_fold_gc = self.fold(x_split_unfold_gc)
        
        ones = torch.ones((batch, channel, height, width)).cuda()
        ones_split_unfold = self.unfold(ones)
        ones_split_fold = self.fold(ones_split_unfold)
       
        out = x_split_fold_gc/ones_split_fold
    
        return out        
      
          

class RAGCBlock(nn.Module):
    def __init__(self,image_size,k, stride, in_channels = 64, out_channels = 64):
        super(RAGCBlock, self).__init__()
        self.image_size = image_size
        self.k = k
        self.stride = stride
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False)
        self.AGCBlock = nn.Sequential(AGCBlock(image_size = self.image_size, k = self.k,  stride = self.stride, padding = 0))

        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.AGCBlock(out)
        out += residual
        return out



class UNet(nn.Module):
    def __init__(self, image_size = 64):
        super(UNet, self).__init__()   
        self.input_channel=1
        self.inter_channel=64
        self.output_channel=64
        self.image_size=image_size
        self.conv1=nn.Sequential(nn.Conv2d(1,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True))
        self.conv2=nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True))
    
        self.layer1=nn.Sequential(RAGCBlock(self.image_size,15, 7,self.inter_channel, self.inter_channel),
                              RAGCBlock(self.image_size,15, 7,self.inter_channel, self.inter_channel),
                              RAGCBlock(self.image_size,15, 7,self.inter_channel, self.inter_channel),
                              RAGCBlock(self.image_size,15, 7,self.inter_channel, self.inter_channel))
    
        self.pool1=nn.MaxPool2d(kernel_size=(2, 2))
    
        self.layer2=nn.Sequential(RAGCBlock(np.int32(self.image_size/2), 8, 2,self.inter_channel, self.inter_channel),
                              RAGCBlock(np.int32(self.image_size/2), 8, 2,self.inter_channel, self.inter_channel),
                              RAGCBlock(np.int32(self.image_size/2), 8, 2,self.inter_channel, self.inter_channel),
                              RAGCBlock(np.int32(self.image_size/2), 8, 2,self.inter_channel, self.inter_channel))
    
        self.pool2=nn.MaxPool2d(kernel_size=(2, 2))
    
        self.layer3=nn.Sequential(RAGCBlock(np.int32(self.image_size/4),8,1,self.inter_channel, self.inter_channel),
                             RAGCBlock(np.int32(self.image_size/4), 8, 2,self.inter_channel, self.inter_channel),
                             RAGCBlock(np.int32(self.image_size/4), 8, 2,self.inter_channel, self.inter_channel),
                             RAGCBlock(np.int32(self.image_size/4), 8, 2,self.inter_channel, self.inter_channel),
                              RAGCBlock(np.int32(self.image_size/4),8,1,self.inter_channel, self.inter_channel),
                              RAGCBlock(np.int32(self.image_size/4),8,1,self.inter_channel, self.inter_channel))
    

        self.pool3=nn.Upsample(scale_factor=2, mode='nearest')
    
        self.layer4=nn.Sequential(RAGCBlock(np.int32(self.image_size/2),8, 2,self.inter_channel, self.inter_channel),
                              RAGCBlock(np.int32(self.image_size/2),8, 2,self.inter_channel, self.inter_channel),
                              RAGCBlock(np.int32(self.image_size/2),8, 2,self.inter_channel, self.inter_channel),
                              RAGCBlock(np.int32(self.image_size/2),8, 2,self.inter_channel, self.inter_channel))
    
    
        self.pool4=nn.Upsample(scale_factor=2, mode='nearest')
    
        self.layer5=nn.Sequential(RAGCBlock(self.image_size,15, 7,self.inter_channel, self.inter_channel),
                              RAGCBlock(self.image_size,15, 7,self.inter_channel, self.inter_channel),
                              RAGCBlock(self.image_size,15, 7,self.inter_channel, self.inter_channel),
                              RAGCBlock(self.image_size,15, 7,self.inter_channel, self.inter_channel))
    

        self.conv3=nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True))
        self.conv4=nn.Sequential(nn.Conv2d(self.inter_channel,1,3,padding=1))
#
    def forward(self,x):
#      
        x=self.conv1(x)
        x2=self.conv2(x)
        x=self.layer1(x2) 
        x4=self.pool1(x)
        x=self.layer2(x4)
        x6=self.pool2(x)
        x7=self.layer3(x6)
        x=self.pool3(x6+x7)
        x9=self.layer4(x)
        x=self.pool4(x9+x4)
        x11=self.layer5(x)
        x= self.conv3(x11+x2)
        x = self.conv4(x) 
        return x

    


class SNConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

        
class ImageDiscriminator(nn.Module):
    def __init__(self, image_size = 64,k = 15,stride = 7,padding = 0, cnum = 64):
        super(ImageDiscriminator, self).__init__()
        self.cnum = cnum
        self.image_size = image_size
        self.k = k
        self.stride = stride
        self.padding = padding
                
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(1, cnum, 3, 1),
            
            SNConvWithActivation(cnum, cnum, 3, 1),
            AGCBlock(image_size = self.image_size, k = self.k, stride = self.stride, padding = self.padding, inplanes = cnum),
           
            SNConvWithActivation(cnum, 2*cnum, 3, 1),
            AGCBlock(image_size = self.image_size, k = self.k, stride = self.stride, padding = self.padding, inplanes = 2*cnum),
            nn.MaxPool2d(kernel_size=(2, 2)),

            
            SNConvWithActivation(2*cnum, 4*cnum, 3, 1),
            AGCBlock(image_size = np.int32(self.image_size/2), k = 8, stride = 2, padding = self.padding, inplanes = 4*cnum),            
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            
            SNConvWithActivation(4*cnum, 8*cnum, 3, 1),
            AGCBlock(image_size = np.int32(self.image_size/4),  k = 8, stride = 2, padding = self.padding, inplanes = 8*cnum), 
            
            
            SNConvWithActivation(8*cnum, 8*cnum, 3, 1),
            AGCBlock(image_size = np.int32(self.image_size/4),  k = 8, stride = 2, padding = self.padding, inplanes = 8*cnum), 
            
            SNConvWithActivation(8*cnum, 8*cnum, 3, 1),
            AGCBlock(image_size = np.int32(self.image_size/4),  k = 8, stride = 2, padding = self.padding, inplanes = 8*cnum)

        )
    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        return x        