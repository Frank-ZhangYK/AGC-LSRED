import argparse
from loader import get_loader
from networks import UNet, ImageDiscriminator
from loss import SNDisLoss, SNGenLoss, NCMSE, p_loss
import torch
import time
import torch.nn as nn



parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--load_mode', type=int, default=1)
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument('--saved_path', type=str, default='./npy_img/')
parser.add_argument('--save_path', type=str, default='./save/') #save_trai
parser.add_argument('--test_patient', type=str, default='L506')

parser.add_argument('--norm_range_min', type=float, default=-1024.0)
parser.add_argument('--norm_range_max', type=float, default=3072.0)
parser.add_argument('--trunc_min', type=float, default=-160.0)
parser.add_argument('--trunc_max', type=float, default=240.0)

parser.add_argument('--transform', type=bool, default=False)
# if patch training, batch size is (--patch_n * --batch_size)
parser.add_argument('--patch_n', type=int, default=20)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--print_iters', type=int, default=20)
parser.add_argument('--decay_iters', type=int, default=6000)
parser.add_argument('--save_iters', type=int, default=1000)
parser.add_argument('--test_iters', type=int, default=1000)


parser.add_argument('--lr', type=float, default=1e-4)

parser.add_argument('--device', type=str)
parser.add_argument('--num_workers', type=int, default=7)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--load_chkpt', type=bool, default=False)

args = parser.parse_args()

data_loader = get_loader(mode=args.mode,
                         load_mode=args.load_mode,
                         saved_path=args.saved_path,
                         test_patient=args.test_patient,
                         patch_n=(args.patch_n if args.mode=='train' else None),
                         patch_size=(args.patch_size if args.mode=='train' else None),
                         transform=args.transform,
                         batch_size=(args.batch_size if args.mode=='train' else 1),
                         num_workers=args.num_workers)


if args.load_chkpt:
    print('Loading Chekpoint')
    whole_model = torch.load(args.save_path+ 'latest_ckpt.pth.tar')
    netG_state_dict,optG_state_dict = whole_model['netG_state_dict'], whole_model['optG_state_dict']
    netD_state_dict,optD_state_dict = whole_model['netD_state_dict'], whole_model['optD_state_dict']
    netG = UNet()
    netG = netG.cuda()
    netD= ImageDiscriminator()
    netD=netD.cuda()
    optG = torch.optim.Adam(netG.parameters())
    optD= torch.optim.Adam(netD.parameters())
    netD.load_state_dict(netD_state_dict)
    netG.load_state_dict(netG_state_dict)
    optG.load_state_dict(optG_state_dict)
    optD.load_state_dict(optD_state_dict)
    cur_epoch = whole_model['epoch']
    total_iters = whole_model['total_iters']
    lr = whole_model['lr']
    print(cur_epoch)
else:
    print('Training model from scrath')
    netG = UNet()
    netG = netG.cuda()
    netD= ImageDiscriminator()
    netD=netD.cuda()
    optG = torch.optim.Adam(netG.parameters(), lr=args.lr)
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr)
    cur_epoch = 0
    total_iters = 0
    lr=args.lr
    
train_losses = []
criterion= NCMSE()
L1_loss = nn.L1Loss()
P_loss=p_loss().cuda()
GANLoss=SNGenLoss()
DLoss=SNDisLoss()
criterion = criterion.cuda()
start_time = time.time()
for epoch in range(cur_epoch, args.num_epochs):
    
    netG.train()
    
    
    for iter_, (x, y) in enumerate(data_loader):
        total_iters += 1
        
        # add 1 channel
        x = x.unsqueeze(0).float()
        y = y.unsqueeze(0).float()
        if args.patch_size:     #patch training
            x = x.view(-1, 1, args.patch_size, args.patch_size)
            y = y.view(-1, 1, args.patch_size, args.patch_size)

        x = x.cuda()
        y = y.cuda()

        pred = netG(x)
        
        
        pos_neg_imgs = torch.cat([y, pred], dim=0)
       
        
        pred_pos_neg = netD(pos_neg_imgs)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
        
        
        d_loss = DLoss(pred_pos, pred_neg, pred, x)
        optD.zero_grad()
        netD.zero_grad()
     
        g_loss = GANLoss(pred,y)

        r_loss= criterion(pred, y, x)

        loss = L1_loss(pred, y) + 2e-1*(P_loss(x, y,'latest_ckpt_pretrain.pth.tar')) + 2e-1*g_loss        
        
        optG.zero_grad()
        netG.zero_grad()

        d_loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        
        
        optD.step()   
        optG.step()        
        
 

            
        # print
        if total_iters % args.print_iters == 0:
            print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nR_LOSS: {:.8f}, G_LOSS: {:.16f}, D_LOSS: {:.16f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        args.num_epochs, iter_+1, 
                                                                                                        len(data_loader), loss.item(), g_loss.item(), d_loss.item()
                                                                                                        ,time.time() - start_time))
        if total_iters % args.decay_iters == 0:
            lr = lr * 0.5
            for param_group in optG.param_groups:
                param_group['lr'] = lr
            for param_group in optD.param_groups:
                param_group['lr'] = 4*lr
        # save model
        if total_iters % args.save_iters == 0:
            saved_model = {
                   'epoch': epoch ,
                   'netG_state_dict': netG.state_dict(),
                   'optG_state_dict': optG.state_dict(),
                   'netD_state_dict': netD.state_dict(),
                   'optD_state_dict': optD.state_dict(),
                   'lr': lr,
                   'total_iters': total_iters
                   }
            torch.save(saved_model, '{}iter_{}_ckpt.pth.tar'.format(args.save_path, total_iters))
            torch.save(saved_model, '{}latest_ckpt.pth.tar'.format(args.save_path))