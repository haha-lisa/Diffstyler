# Diffstyler
DiffStyler: Controllable Dual Diffusion for Text-Driven Image Stylization

## Official Pytorch implementation of "Draw Your Art Dream: Diverse Digital Art Synthesis with Multimodal Guided Diffusion" 

[Project Page(https://arxiv.org/abs/2211.10682)]

![MAIN3_e2-min](https://github.com/haha-lisa/Diffstyler/blob/main/teaser10.png)

[Nisha Huang], [Yuxin Zhang], [Fan Tang], [Chongyang Ma], [Haibin Huang], [Yong Zhang], [Weiming Dong], [Changsheng Xu]

## Abstract
> Despite the impressive results of arbitrary image-guided style transfer methods, text-driven image stylization has recently been proposed for transferring a natural image into the stylized one according to textual descriptions of the target style provided by the user. Unlike previous image-to-image transfer approaches, text-guided stylization progress provides users with a more precise and intuitive way to express the desired style. However, the huge discrepancy between cross-modal inputs/outputs makes it challenging to conduct text-driven image stylization in a typical feed-forward CNN pipeline. In this paper, we present DiffStyler on the basis of diffusion models. The cross-modal style information can be easily integrated as guidance during the diffusion progress step-by-step. In particular, we use a dual diffusion processing architecture to control the balance between the content and style of the diffused results. Furthermore, we propose a content image-based learnable noise on which the reverse denoising process is based, enabling the stylization results to better preserve the structure information of the content image. We validate the proposed DiffStyler beyond the baseline methods through extensive qualitative and quantitative experiments.


## Cite
```
@inproceedings{Huang2022DIFFSTYLER,
author = {Nisha, Huang and Yuxin, Zhang and Fan, Tang and Chongyang, Ma and Haibin, Huang and Yong, Zhang and Weiming, Dong and Changsheng, Xu},
title = {DiffStyler: Controllable Dual Diffusion for Text-Driven Image Stylization},
year = {2022}
}
```
