#!/usr/bin/env python3

"""Applies a text prompt to an existing image by finding a latent that would produce it
with the unconditioned DDIM ODE, then integrating the text-conditional DDIM ODE starting
from that latent."""

import argparse
from functools import partial
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from torchvision.transforms import functional as TF
from tqdm import trange

from CLIP import clip
from diffusion import get_model, get_models, sampling, utils

import networks
import net
from packaging import version

import lpips

MODULE_DIR = Path(__file__).resolve().parent

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def parse_prompt(prompt, default_weight=3.):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', default_weight][len(vals):]
    return vals[0], float(vals[1])


def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])

############################################## tv_loss #################################################
def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class PatchNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.similarity_function = self._get_similarity_function()
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def _get_similarity_function(self):

        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        return self._cosine_simililarity

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, M, C)
        # v shape: (N, M)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        feat_k = feat_k.detach()
        l_pos = self.cos(feat_q,feat_k)
        l_pos = l_pos.view(batchSize, 1)
        l_neg_curbatch = self.similarity_function(feat_q.view(batchSize,1,-1),feat_k.view(1,batchSize,-1))
        l_neg_curbatch = l_neg_curbatch.view(1,batchSize,-1)
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(batchSize, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, batchSize)
        out = torch.cat((l_pos, l_neg), dim=1) / 0.07
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('init', type=str,
                   help='the init image')
    p.add_argument('prompts', type=str, default=[], nargs='*',
                   help='the text prompts to use')
    p.add_argument('--images', type=str, default=[], nargs='*', metavar='IMAGE',
                   help='the image prompts')
    p.add_argument('--checkpoint', type=str,
                   help='the checkpoint to use')
    p.add_argument('--device', type=str, 
                   help='the device to use')
    p.add_argument('--max-timestep', '-mt', type=float, default=1.,
                   help='the maximum timestep')
    p.add_argument('--method', type=str, default='iplms',
                   choices=['ddim', 'prk', 'plms', 'pie', 'plms2', 'iplms'],
                   help='the sampling method to use')
    p.add_argument('--model', type=str, default='cc12m_1_cfg', choices=['cc12m_1_cfg'],
                   help='the model to use')
    p.add_argument('--output', '-o', type=str, default='out.png',
                   help='the output filename')
    p.add_argument('--size', type=int, nargs=2,
                   help='the output image size')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of timesteps')

    
    p.add_argument('--cutn', type=int, default=1,
                   help='the number of random crops to use')
    p.add_argument('--cut-pow', type=float, default=1.,
                   help='the random crop size power')
    p.add_argument('--clip-guidance-scale', '-cs', type=float, default=500.,
                   help='the CLIP guidance scale')
    p.add_argument('-n', type=int, default=1,
                   help='the number of images to sample')
    p.add_argument('--checkpoint1', type=str,
                   help='the checkpoint to use')
    p.add_argument('--model1', type=str, default='wikiart_256', choices=get_models(),
                   help='the model to use')
    p.add_argument('--wikiart_scale', '-ws', type=float, default=0.5,
                   help='wikiart_scale')
    p.add_argument('--free_scale', '-fs', type=float, default=0.5,
                   help='cfg_scale')

    p.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')


    p.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    p.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
    p.add_argument('--no_dropout', type=str2bool, nargs='?', const=True, default=True,
                            help='no dropout for the generator')
    p.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
    p.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    p.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
    p.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    p.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
    p.add_argument('--content_nce_layers', type=str, default='1,2,3,4', help='compute NCE loss on which layers')
    # p.add_argument('--content_nce_layers', type=str, default='1,2,3,4', help='compute NCE loss on which layers')
    p.add_argument('-ns', '--nce_scale', type=float, default=1., help='the nce loss scale')

    p.add_argument("-lc", '--lambda_c', type=float, default=3.,
                    help='content loss parameter')


    p.add_argument("-tvs",  "--tv_scale", type=float, help="Smoothness scale", default=0, dest='tv_scale') 
    p.add_argument("-is",   "--init_scale", type=int, help="Initial image scale (e.g. 1000)", default=0, dest='init_scale') 
    p.add_argument("-as",   "--aes_scale", type=float, default=0., help='aesthetic_loss_scale')

    args = p.parse_args()


    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    vgg = net.vgg
    # self.netAE = net.ADAIN_Encoder(vgg, self.gpu_ids)
    netAE = net.ADAIN_Encoder(vgg, args.gpu_ids).to(device)
    netF = networks.define_F(args.input_nc, 'mlp_sample', args.normG,
                                       not args.no_dropout, args.init_type, args.init_gain, args.no_antialias, args.gpu_ids).to(device)
    
    VGG = models.vgg19(pretrained=True).features
    VGG.to(device)


    criterionNCE = []
    for nce_layer in args.content_nce_layers:
                criterionNCE.append(PatchNCELoss().to(device))

    model = get_model(args.model)()
    _, side_y, side_x = model.shape
    if args.size:
        side_x, side_y = args.size
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = MODULE_DIR / f'checkpoints/{args.model}.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    if device.type == 'cuda':
        model = model.half()
    model = model.to(device).eval().requires_grad_(False)
    clip_model_name = model.clip_model if hasattr(model, 'clip_model') else 'ViT-B/16'
    clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    lpips_model = lpips.LPIPS(net='vgg').to(device)

    init = Image.open(utils.fetch(args.init)).convert('RGB')
    init = resize_and_center_crop(init, (side_x, side_y))
    init = utils.from_pil_image(init).to(device)[None]

    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, args.cutn, args.cut_pow)

    zero_embed = torch.zeros([1, clip_model.visual.output_dim], device=device)
    target_embeds, weights = [zero_embed], []

    target_embeds1, weights1 = [], []
    for prompt in args.prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)
        target_embeds1.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights1.append(weight)

    for prompt in args.images:
        path, weight = parse_prompt(prompt)
        img = Image.open(utils.fetch(path)).convert('RGB')
        img1 = TF.resize(img, min(side_x, side_y, *img.size),
                        transforms.InterpolationMode.LANCZOS)
        clip_size = clip_model.visual.input_resolution
        img = resize_and_center_crop(img, (clip_size, clip_size))
        batch = TF.to_tensor(img)[None].to(device)
        embed = F.normalize(clip_model.encode_image(normalize(batch)).float(), dim=-1)
        target_embeds.append(embed)
        weights.append(weight)

         
        batch1= make_cutouts(TF.to_tensor(img1)[None].to(device))
        embeds1 = F.normalize(clip_model.encode_image(normalize(batch1)).float(), dim=-1)
        target_embeds1.append(embeds1)
        weights1.extend([weight / args.cutn] * args.cutn)

    weights = torch.tensor([1 - sum(weights), *weights], device=device)


    target_embeds1 = torch.cat(target_embeds1)
    weights1 = torch.tensor(weights1, device=device)
    if weights1.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights1 /= weights1.sum().abs()
    





    model1 = get_model(args.model1)()
    _, side_y, side_x = model1.shape
    # if args.size:
    #     side_x, side_y = args.size
    checkpoint1 = args.checkpoint1
    if not checkpoint1:
        checkpoint1 = MODULE_DIR / f'checkpoints/{args.model1}.pth'
    model1.load_state_dict(torch.load(checkpoint1, map_location='cpu'))
    if device.type == 'cuda':
        model1 = model1.half()
    model1 = model1.to(device).eval().requires_grad_(False)

    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, args.cutn, args.cut_pow)


    aesthetic_model_16 = torch.nn.Linear(512,1).cuda()
    aesthetic_model_16.load_state_dict(torch.load("./checkpoints/ava_vit_b_16_linear.pth"))

    def cond_model_fn(x, t, **extra_args):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            v = model1(x, t, **extra_args)
            alphas, sigmas = utils.t_to_alpha_sigma(t)
            pred = x * alphas[:, None, None, None] - v * sigmas[:, None, None, None]
            cond_grad = cond_fn(x, t, pred, **extra_args).detach()
            v = v.detach() - cond_grad * (sigmas[:, None, None, None] / alphas[:, None, None, None])
        return v



    def calculate_NCE_loss(src, tgt):
        content_nce_layers = [int(i) for i in args.content_nce_layers.split(',')]    

        n_layers = len(content_nce_layers)
        feat_q, feat_k = netAE(tgt, src, encoded_only = True)
        #feat_q = self.netG_B(tgt, self.style_A, self.nce_layers, encode_only=True)
        #feat_k = self.netG_A(src, self.style_B, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = netF(feat_k, args.num_patches, None)
        feat_q_pool, _ = netF(feat_q, args.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, criterionNCE, args.content_nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        # print('total_nce_loss_A',total_nce_loss)
        return total_nce_loss / n_layers

    def img_normalize(image):
        mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
        std=torch.tensor([0.229, 0.224, 0.225]).to(device)
        mean = mean.view(1,-1,1,1)
        std = std.view(1,-1,1,1)

        image = (image-mean)/std
        return image

    def load_image2(img_path, img_height=None,img_width =None):
    
        image = Image.open(img_path)
        if img_width is not None:
            image = image.resize((img_width, img_height))  # change image size to (3, img_size, img_size)
        
        transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])   

        image = transform(image)[:3, :, :].unsqueeze(0)

        return image
    
    def get_features(image, model, layers=None):

        if layers is None:
            layers = {'0': 'conv1_1',  
                    '5': 'conv2_1',  
                    '10': 'conv3_1', 
                    '19': 'conv4_1', 
                    '21': 'conv4_2', 
                    '28': 'conv5_1',
                    '31': 'conv5_2'
                    }  
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)   
            if name in layers:
                features[layers[name]] = x
        
        return features




    def cond_fn(x, t, pred):
        clip_embed = F.normalize(target_embeds1.mul(weights1[:, None]).sum(0, keepdim=True), dim=-1)
        clip_embed = clip_embed.repeat([args.n, 1])
        if min(pred.shape[2:4]) < 256:
            pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=False)
        clip_in = normalize(make_cutouts((pred + 1) / 2))
        image_embeds = clip_model.encode_image(clip_in).view([args.cutn, x.shape[0], -1])
        losses = spherical_dist_loss(image_embeds, clip_embed[None])
        loss_NCE =calculate_NCE_loss(init.detach().to(device), pred.detach().to(device))
        clip_loss = losses.mean(0).sum()

        content_image = load_image2(args.init, 256,256)
        content_image = content_image.to(device)
        content_features = get_features(img_normalize(content_image), VGG)
        # target = 
        target_features = get_features(img_normalize(pred), VGG)
        content_loss = 0
        content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)



        tv_losses = tv_loss(pred)

        init_losses = lpips_model(pred, init)

        aes_loss = (aesthetic_model_16(F.normalize(image_embeds, dim=-1))).mean() 


        total_loss = clip_loss * args.clip_guidance_scale+ loss_NCE * args.nce_scale \
                     + content_loss *args.lambda_c + tv_losses.sum() * args.tv_scale + init_losses.sum() * args.init_scale + aes_loss * args.aes_scale 
        grad = -torch.autograd.grad(total_loss, x)[0]
        return grad

    def cfg_model_fn(x, t):
        n = x.shape[0]
        n_conds = len(target_embeds)
        x_in = x.repeat([n_conds, 1, 1, 1])
        t_in = t.repeat([n_conds])
        clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
        vs = model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
        v = vs.mul(weights[:, None, None, None, None]).sum(0)

        # if hasattr(model1, 'clip_model'):
        #     extra_args = {'clip_embed': clip_embed}
        # else:
        #     extra_args = {}
        # extra_args = {}
        # v1 = cond_model_fn(x, t, **extra_args)
        v1 = cond_model_fn(x, t)
        v = args.free_scale*v+args.wikiart_scale*v1
        return v

    def run():
        t = torch.linspace(0, 1, args.steps + 1, device=device)
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        steps = steps[steps <= args.max_timestep]
        if args.method == 'ddim':
            x = sampling.reverse_sample(model, init, steps, {'clip_embed': zero_embed})
            out = sampling.sample(cfg_model_fn, x, steps.flip(0)[:-1], 0, {})
        if args.method == 'prk':
            x = sampling.prk_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.prk_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        if args.method == 'plms':
            x = sampling.plms_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.plms_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
            # out1 = sampling.plms_sample(cfg_model_fn, x, steps.flip(0)[:-3], {})
            # out2 = sampling.plms_sample(cfg_model_fn, x, steps.flip(20)[:-1], {})
            # out3 = sampling.plms_sample(cfg_model_fn, x, steps.flip(30)[:-1], {})
            # out4 = sampling.plms_sample(cfg_model_fn, x, steps.flip(40)[:-1], {})
            # out5 = sampling.plms_sample(cfg_model_fn, x, steps.flip(50)[:-1], {})
        if args.method == 'pie':
            x = sampling.pie_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.pie_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        if args.method == 'plms2':
            x = sampling.plms2_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.plms2_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        if args.method == 'iplms':
            x = sampling.iplms_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.iplms_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        utils.to_pil_image(out[0]).save(args.output)
        # utils.to_pil_image(out[0]).save("./output/ablation/steps/cat-step50.png")
        # utils.to_pil_image(out1[0]).save("./output/ablation/steps/cat-step40.png")
        # utils.to_pil_image(out2[0]).save("./output/ablation/steps/cat-step30.png")
        # utils.to_pil_image(out3[0]).save("./output/ablation/steps/cat-step20.png")
        # utils.to_pil_image(out4[0]).save("./output/ablation/steps/cat-step10.png")
        # utils.to_pil_image(out5[0]).save("./output/ablation/steps/cat-step00.png")
    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
