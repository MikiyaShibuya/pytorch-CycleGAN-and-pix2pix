import torch
import itertools
from .base_model import BaseModel
from . import networks
from util import util


class TransModalModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='paired', input_nc=1)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_Cycle', type=float, default=100.0, help='weight for Cycle consistency loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_A', 'G_B', 'G_A_L1', 'G_B_L1', 'D_AB_real', 'D_AB_fake', 'D_BA_real', 'D_BA_fake', 'Cycle_A', 'Cycle_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A_viz', 'fake_A_viz', 'rec_A_viz', 'real_B', 'fake_B', 'rec_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_AB', 'D_BA']
        else:  # during test time, only load G
            self.model_names = ['G_A', 'G_B']
        # define networks (both generator and discriminator)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD_AB = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_BA = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_AB.parameters(), self.netD_BA.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A_viz = self.fake_fir(self.real_A)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def fake_fir(self, fir_tensor):
        fir = fir_tensor.clone()
        fir.detach()
        return (torch.clamp(fir, -0.8, 0.2) + 0.3) * 2

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

        self.fake_A_viz = self.fake_fir(self.fake_A)
        self.rec_A_viz = self.fake_fir(self.rec_A)

    def backward_D_A(self, no_backward=False):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake_AB = self.netD_AB(fake_AB.detach())
        self.loss_D_AB_fake = self.criterionGAN(pred_fake_AB, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real_AB = self.netD_AB(real_AB)
        self.loss_D_AB_real = self.criterionGAN(pred_real_AB, True)

        # combine loss and calculate gradients
        if not no_backward:
            loss_D = (self.loss_D_AB_fake + self.loss_D_AB_real) * 0.5
            loss_D.backward()

    def backward_D_B(self, no_backward=False):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_BA = torch.cat((self.real_B, self.fake_A), 1)
        pred_fake_BA = self.netD_BA(fake_BA.detach())
        self.loss_D_BA_fake = self.criterionGAN(pred_fake_BA, False)

        # Real
        real_BA = torch.cat((self.real_B, self.real_A), 1)
        pred_real_BA = self.netD_BA(real_BA)
        self.loss_D_BA_real = self.criterionGAN(pred_real_BA, True)
        # combine loss and calculate gradients
        if not no_backward:
            loss_D = (self.loss_D_BA_fake + self.loss_D_BA_real) * 0.5
            loss_D.backward()

    def backward_G(self, no_backward=False):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake_AB = self.netD_AB(fake_AB)
        self.loss_G_A = self.criterionGAN(pred_fake_AB, True)

        fake_BA = torch.cat((self.real_B, self.fake_A), 1)
        pred_fake_BA = self.netD_BA(fake_BA)
        self.loss_G_B = self.criterionGAN(pred_fake_BA, True)

        # Second, G(A) = B
        self.loss_G_A_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_B_L1 = self.criterionL1(self.fake_A, self.real_A) * self.opt.lambda_L1

        self.loss_Cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_Cycle_A
        self.loss_Cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_Cycle_B

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B \
                      + self.loss_G_A_L1 + self.loss_G_B_L1 \
                      + self.loss_Cycle_A + self.loss_Cycle_B
        if not no_backward:
            self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD_AB, True)  # enable backprop for D
        self.set_requires_grad(self.netD_BA, True)
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D_A()              # calculate gradients for D
        self.backward_D_B()
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD_AB, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_BA, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def test(self):
        super().test()  # forward() method is called by this

        real_A = util.tensor2np(self.real_A)
        fake_A = util.tensor2np(self.fake_A.detach())
        psnr_A = self.PSNR(real_A, fake_A, 2.)
        ssim_A = self.SSIM(real_A, fake_A, multichannel=len(real_A.shape)==3, data_range=2)

        real_B = util.tensor2np(self.real_B)
        fake_B = util.tensor2np(self.fake_B.detach())
        psnr_B = self.PSNR(real_B, fake_B, 2.)
        ssim_B = self.SSIM(real_B, fake_B, multichannel=len(real_B.shape)==3, data_range=2)

        return {'PSNR_A': psnr_A, 'PSNR_B': psnr_B, 'SSIM_A': ssim_A, 'SSIM_B': ssim_B}

    def validation(self):
        '''
        Do validation, no parameter optimization, only computing losses.
        :return:
        '''
        self.set_requires_grad(self.netD_AB, False)
        self.set_requires_grad(self.netD_BA, False)
        self.set_requires_grad(self.netG_A, False)
        self.set_requires_grad(self.netG_B, False)
        self.backward_D_A(no_backward=True)
        self.backward_D_B(no_backward=True)
        self.backward_G(no_backward=True)
        self.set_requires_grad(self.netG_A, True)
        self.set_requires_grad(self.netG_B, True)
