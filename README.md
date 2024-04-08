# Diffstyler: Controllable Dual Diffusion for Text-Driven Image Stylization
DiffStyler: Controllable Dual Diffusion for Text-Driven Image Stylization

## Official Pytorch implementation of ["DiffStyler: Controllable Dual Diffusion for Text-Driven Image Stylization"](https://arxiv.org/abs/2211.10682)

[Project Page(https://arxiv.org/abs/2211.10682)]

![MAIN3_e2-min](https://github.com/haha-lisa/Diffstyler/blob/main/figure/teaser2.jpg)


## Abstract
> Despite the impressive results of arbitrary image-guided style transfer methods, text-driven image stylization has recently been proposed for transferring a natural image into the stylized one according to textual descriptions of the target style provided by the user. Unlike previous image-to-image transfer approaches, text-guided stylization progress provides users with a more precise and intuitive way to express the desired style. However, the huge discrepancy between cross-modal inputs/outputs makes it challenging to conduct text-driven image stylization in a typical feed-forward CNN pipeline. In this paper, we present DiffStyler on the basis of diffusion models. The cross-modal style information can be easily integrated as guidance during the diffusion progress step-by-step. In particular, we use a dual diffusion processing architecture to control the balance between the content and style of the diffused results. Furthermore, we propose a content image-based learnable noise on which the reverse denoising process is based, enabling the stylization results to better preserve the structure information of the content image. We validate the proposed DiffStyler beyond the baseline methods through extensive qualitative and quantitative experiments.

## Framework
![MAIN3_e2-min](https://github.com/haha-lisa/Diffstyler/blob/main/figure/pipeline5.jpg)
The overall pipeline of our DiffStyler framework.
## Environment
```
conda create -n diffstyler python=3.9
conda activate diffstyler
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Install dependencies
```
pip install -e ./CLIP
pip install -r requirements.txt
```

## Pretrained models
[CC12M_cfg](https://the-eye.eu/public/AI/models/v-diffusion/cc12m_1_cfg.pth), [WikiArt](https://the-eye.eu/public/AI/models/v-diffusion/wikiart_256.pth)
<br> Please download them and put them into the floder ./checkpoints/ <br> 

## Run
```
python main.py "dog.jpg" "An oil painting of a dog in impressionism style." --output "test_dog.png" -fs 0.8 -ws 0.2 -lc 3 --steps 50
```


## License
The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file.<be>

## Citation
If you find our work is useful in your research, please consider citing:

```
@article{huang2024diffstyler,
  title={Diffstyler: Controllable dual diffusion for text-driven image stylization},
  author={Huang, Nisha and Zhang, Yuxin and Tang, Fan and Ma, Chongyang and Huang, Haibin and Dong, Weiming and Xu, Changsheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```
