import torch
import torch.nn as nn
import torch.nn.parallel


class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:relu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
        output = torch.exp(output)
        output = output.mean(0)
        return output.view(1)


class DCGAN_Dd(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_Dd, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"


        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)


        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

        


    def forward(self, input):
        #print(input.shape)
        conv1 = self.conv1(input)
        #print(conv1.shape)
        relu1 = self.relu1(conv1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)

        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.relu3(bn3)

        conv4 = self.conv4(relu3)
        bn4 = self.bn4(conv4)
        relu4 = self.relu4(bn4)

        conv5 = self.conv5(relu4)
        
        output = conv5
        output = torch.exp(output)
        output = output.mean(0)
        #print(output.shape)
        return (output.view(1),conv4) # changed from conv4


class DCGAN_Dd_noBN(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_Dd_noBN, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"


        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        #self.bn2 = nn.BatchNorm2d(128,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.bn2 = nn.LayerNorm([128,16,16])  # layer norm with name bn
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        #self.bn3 = nn.BatchNorm2d(256,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.bn3 = nn.LayerNorm([256,8,8])
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        #self.bn4 = nn.BatchNorm2d(512,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.bn4 = nn.LayerNorm([512,4,4])
        self.relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)


        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

        


    def forward(self, input):
        #print(input.shape)
        conv1 = self.conv1(input)
        #print(conv1.shape)
        relu1 = self.relu1(conv1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(conv2)

        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.relu3(conv3)

        conv4 = self.conv4(relu3)
        bn4 = self.bn4(conv4)
        #print (bn4.shape)
        relu4 = self.relu4(conv4)

        conv5 = self.conv5(relu4)
        
        output = conv5
        #print(output.shape)
        output = torch.exp(output)
        output = output.mean(0)
        #print(output.shape)
        return (output.view(1),conv4) # change

class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid:{0}:relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
        return output 


class DCGAN_Gd(nn.Module):

    def __init__(self, isize, nz, nc, ngf, ngpu,discr, n_extra_layers=0):
        super(DCGAN_Gd, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.conv1 = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(512,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.tanh5 = nn.Tanh()

        self.query = nn.Conv2d(512, 512, 1, 1, 0, bias=False)
        self.bn_q = nn.BatchNorm2d(512,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.key = nn.Conv2d(512, 512, 1, 1, 0, bias=False)
        self.bn_k = nn.BatchNorm2d(512,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)
        self.value = nn.Conv2d(512, 512, 1, 1, 0, bias=False)
        self.bn_v = nn.BatchNorm2d(512,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)

        self.fin_conv = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)

        self.bn_fd = nn.BatchNorm2d(512,eps=1e-05,momentum = 0.1,affine= True,track_running_stats= True)

        self.softmax  = nn.Softmax(dim=-1) #check why this function is necessary
    #def forward(self,input,disc):
        # only get the 4th layer of disc as input for the gen part
    def forward(self,input,disc=None,use_feedback=False):
        if not use_feedback:
            conv1 = self.conv1(input)
            bn1 = self.bn1(conv1)
            relu1 = self.relu1(bn1)

            conv2 = self.conv2(relu1)
            bn2 = self.bn2(conv2)
            relu2 = self.relu2(bn2)

            conv3 = self.conv3(relu2)
            bn3 = self.bn3(conv3)
            relu3 = self.relu3(bn3)

            conv4 = self.conv4(relu3)
            bn4 = self.bn4(conv4)
            relu4 = self.relu4(bn4)

            conv5 = self.conv5(relu4)
            tanh5 = self.tanh5(conv5)
        
        else:
            
            conv1 = self.conv1(input)
            bn1 = self.bn1(conv1)
            relu1 = self.relu1(bn1)


            disc = self.bn_fd(disc)

            query =  self.bn_q(self.query(relu1)).view(-1,512,4*4).permute(0,2,1)

            
            
            key = self.bn_k(self.key(disc)).view(-1,512,4*4)
            #print(query.shape)
            #print(key.shape)
            mult = self.softmax(torch.bmm(query,key))

            #print(mult.shape)
            value = self.bn_v(self.value(disc)).view(-1,512,4*4).permute(0,2,1)

            #print(mult.shape)
            #print(value.shape)
            out = torch.bmm(mult,value).view(-1,512,4,4)

           
            
            out = self.fin_conv(out)
            #print(out.shape)
            conv2 = self.conv2(relu1 )+ out


           
            bn2 = self.bn2(conv2)
            relu2 = self.relu2(bn2)

            conv3 = self.conv3(relu2)
            bn3 = self.bn3(conv3)
            relu3 = self.relu3(bn3)

            conv4 = self.conv4(relu3)
            bn4 = self.bn4(conv4)
            relu4 = self.relu4(bn4)

            conv5 = self.conv5(relu4)
            tanh5 = self.tanh5(conv5)

            


        return tanh5

###############################################################################
class DCGAN_D_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:conv'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
            
        output = output.mean(0)
        return output.view(1)

class DCGAN_G_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial:{0}-{1}:convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,  range(self.ngpu))
        else: 
            output = self.main(input)
        return output 
