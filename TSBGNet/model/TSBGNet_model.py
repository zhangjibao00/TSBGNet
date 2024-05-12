import torch
from .base_model import BaseModel
import torch.nn.functional as F

from . import network, base_function, external_function
from util import task
import itertools

class GLIIM(BaseModel):
    """This class implements the pluralistic image completion, for 256*256 resolution image inpainting"""
    def name(self):
        return "Pluralistic Image Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale_s', type=int, default=4, help='# of number of the output scale')
        parser.add_argument('--output_scale_inp', type=int, default=5, help='# of number of the output scale')
        if is_train:
            parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_g', type=float, default=1, help='weight for generation loss')

        return parser

    def __init__(self, opt):
        """Initial the pluralistic model"""
        BaseModel.__init__(self, opt)

        self.visual_names = ['img_m', 'img_truth', 'img_out_inp']
        self.value_names = ['u_m', 'sigma_m', 'u_prior', 'sigma_prior']
        self.model_names = [ 'inpainting', 'D2']
        self.loss_names = ['inp',  'app_inpainting', 'ad_inpainting',  'img_inp']
        self.distribution = []


        #self.net_structure = network.define_structure(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_inpainting = network.define_inpainting(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        #self.net_D1 = network.define_Discriminator_1(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_D2 = network.define_Discriminator_2(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.lossNet = network.VGG16FeatureExtractor()
        self.lossNet.cuda(opt.gpu_ids[0])


        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                                                filter(lambda p: p.requires_grad, self.net_inpainting.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                                                                filter(lambda p: p.requires_grad, self.net_D2.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # load the pretrained model and schedulers
        self.setup(opt)

    def set_input(self, input):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.mask = input['mask']



        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])


        # get I_m and I_c for image with mask and complement regions for training
        self.img_truth = self.img * 2 - 1
        self.img_m = (1 - self.mask) * self.img_truth + self.mask

        # get multiple scales image ground truth and mask for training
        self.scale_img2 = []
        self.scale_mask2 = []
        # for i in range(3):
        #
        #     a = F.interpolate(input=self.img_truth, size=(66, 66), mode='bilinear')
        #     self.scale_img2.append(a)
        # self.scale_img2.append(self.img_truth)
        # for i in range(3):
        #
        #     a = F.interpolate(input=self.mask, size=(66, 66), mode='bilinear')
        #     self.scale_mask2.append(a)
        # self.scale_mask2.append(self.mask)
        self.scale_img2 = task.scale_pyramid(self.img_truth, self.opt.output_scale_inp)
        self.scale_mask2 = task.scale_pyramid(self.mask, self.opt.output_scale_inp)

    def test(self):
        """Forward function used in test time"""
        # save the groundtruth and masked image
        self.save_results(self.img_truth, data_name='truth')
        self.save_results(self.img_m, data_name='mask')


        # encoder process
        self.image = torch.cat([self.img_m, self.mask], dim=1)

        inp_feature, inp_result  = self.net_inpainting(self.image)

        self.img_out_inp = inp_result[-1]
        self.save_results(self.img_out_inp, data_name='out')


    def forward(self):
        """Run forward processing to get the inputs"""
        # encoder process
        self.image = torch.cat([self.img_m, self.mask], dim=1)
        #s_feature, s_result = self.net_structure(self.image)
        inp_feature, inp_result = self.net_inpainting(self.image)

        self.img_inp = []
        for result in inp_result:
            img_inp = result
            self.img_inp.append(img_inp)

        self.img_out_inp = self.img_inp[-1].detach()


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # global
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 10

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        #base_function._unfreeze(self.net_D1)
        base_function._unfreeze(self.net_D2)
        #self.loss_img_s = self.backward_D_basic(self.net_D1, self.img_structure, self.img_s[-1])
        self.loss_img_inp = self.backward_D_basic(self.net_D2, self.img_truth, self.img_inp[-1])

    def backward_G(self):
        """Calculate training loss for the generator"""
        # generator adversarial loss
        #base_function._freeze(self.net_D1)
        base_function._freeze(self.net_D2)
        # D_fake_structure = self.net_D1(self.img_s[-1])
        # D_real_structure = self.net_D1(self.img_structure)
        #self.loss_ad_structure = self.L2loss(D_fake_structure, D_real_structure) * self.opt.lambda_g
        D_fake_inpainting = self.net_D2(self.img_inp[-1])
        D_real_inpainting = self.net_D2(self.img_truth)
        self.loss_ad_inpainting = self.L2loss(D_fake_inpainting, D_real_inpainting) * self.opt.lambda_g

        # calculate l1 loss for multi-scale outputs
        loss_app_hole_s, loss_app_context_s, loss_app_hole_inp, loss_app_context_inp = 0,0,0,0
        for i, (img_fake_i, img_real_i, mask_i) in enumerate(zip(self.img_inp, self.scale_img2, self.scale_mask2)):
            loss_app_hole_inp += self.L1loss(img_fake_i*mask_i, img_real_i*mask_i)
            loss_app_context_inp += self.L1loss(img_fake_i * (1-mask_i), img_real_i * (1-mask_i))

        #self.loss_app_structure = loss_app_hole_s * self.opt.lambda_rec + loss_app_context_s * self.opt.lambda_rec
        self.loss_app_inpainting = loss_app_hole_inp * self.opt.lambda_rec + loss_app_context_inp * self.opt.lambda_rec

        real_feats2 = self.lossNet(self.img_truth)
        fake_feats2 = self.lossNet(self.img_out_inp)

        self.loss_inp_style = base_function.style_loss(real_feats2, fake_feats2)
        self.loss_inp_content = base_function.perceptual_loss(real_feats2, fake_feats2)
        self.loss_inp = self.loss_inp_style + 10*self.loss_inp_content

        total_loss = 0

        for name in self.loss_names:
            if name != 'img_s' and name != 'img_inp':
                total_loss += getattr(self, "loss_" + name)

        total_loss.backward()


    def optimize_parameters(self):
        """update network weights"""
        # compute the image completion results
        self.forward()
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
