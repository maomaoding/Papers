# Papers 

## Fundamental related
[On Calibration of Modern Neural Networks](https://arxiv.org/pdf/1706.04599.pdf)  

## Material Classification
[Differential Viewpoints for Ground Terrain Material Recognition](https://arxiv.org/pdf/2009.11072v1.pdf)  
[Exploring Features in a Bayesian Framework for Material Recognition](http://people.csail.mit.edu/celiu/CVPR2010/maltRecogCVPR10.pdf)  
[Toward Robust Material Recognition for Everyday Objects](https://homes.cs.washington.edu/~xren/publication/hu-bmvc11-material.pdf)  
[Recognizing Material Properties from Images](https://arxiv.org/pdf/1801.03127.pdf)  
[Texture Classification using Block Intensity and Gradient Difference Descriptor](https://arxiv.org/pdf/2002.01154.pdf)  
[Deep Structure-Revealed Network for Texture Recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhai_Deep_Structure-Revealed_Network_for_Texture_Recognition_CVPR_2020_paper.pdf)  
[Material Recognition in the Wild with the Materials in Context Database](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.837.3477&rep=rep1&type=pdf)  
[Reflectance Hashing for Material Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Reflectance_Hashing_for_2015_CVPR_paper.pdf)  
[Deep Texture Manifold for Ground Terrain Recognition](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf)  

## Pytorch Internals
[Pytorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)  
[Pytorch Core Code Research](https://www.miracleyoo.com/2019/12/11/Pytorch-Core-Code-Research/)

## Scene Parsing
[Scene Parsing through ADE20K Dataset](http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

## Semantic Segmentation
[Contour Detection and Hierarchical Image Segmentation](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)

## Instance Segmentation
[PlaneRCNN: 3D Plane Detection and Reconstruction from a Single Image](https://arxiv.org/pdf/1812.04072.pdf)

## Locate Vanishing Point
[Detecting Dominant Vanishing Points in Natural
Scenes with Application to Composition-Sensitive
Image Retrieval](https://arxiv.org/pdf/1608.04267.pdf)
[NeurVPS: Neural Vanishing Point Scanning via Conic Convolution](https://arxiv.org/pdf/1910.06316.pdf)
[2-Line Exhaustive Searching for Real-Time Vanishing Point Estimation in Manhattan World](https://xiaohulugo.github.io/papers/Vanishing_Point_Detection_WACV2017.pdf)
[Rolling Shutter Correction in Manhattan World](https://openaccess.thecvf.com/content_ICCV_2017/papers/Purkait_Rolling_Shutter_Correction_ICCV_2017_paper.pdf)
[Finding Vanishing Points via Point Alignments in Image Primal and Dual Domains](https://openaccess.thecvf.com/content_cvpr_2014/papers/Lezama_Finding_Vanishing_Points_2014_CVPR_paper.pdf)
[3-line RANSAC for Orthogonal Vanishing Point Detection](https://people.inf.ethz.ch/pomarc/pubs/BazinIROS12.pdf)

## Distributed training
[BAGUA: Scaling up Distributed Learning with System Relaxations](https://arxiv.org/pdf/2107.01499.pdf)  

## Camera Calibration
[Camera Calibration with Lens Distortion from Low-rank Textures](https://people.eecs.berkeley.edu/~yima/matrix-rank/Files/calibration.pdf)
[Unsupervised intrinsic calibration from a single frame using a ”plumb-line”approach](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Melo_Unsupervised_Intrinsic_Calibration_2013_ICCV_paper.pdf)
[Simultaneous Vanishing Point Detection and Camera Calibration from Single Images](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.691.5867&rep=rep1&type=pdf)  

## Transformer Related
[AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)  
[LSTR :End-to-end Lane Shape Prediction with Transformers](https://openaccess.thecvf.com/content/WACV2021/papers/Liu_End-to-End_Lane_Shape_Prediction_With_Transformers_WACV_2021_paper.pdf)  
[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)  
* Feature Pyramid
* Local transformer(changing order of magnitude) + shifted window
* Relative Position Encoding  

[Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction
without Convolutions](https://arxiv.org/pdf/2102.12122.pdf)  
* Feature Pyramid
* Spatial Reduction(same order of magnitude)  

[CROSSFORMER: A VERSATILE VISION TRANSFORMER
HINGING ON CROSS-SCALE ATTENTION](https://arxiv.org/pdf/2108.00154.pdf)  
* cross-scale embedding layer(CEL) functions just like spatial pyramid pooling layer(SPP)  
* long short distance attention(LSDA) functions just like dilated convolution to reduce complexity of the vanilla transformer module  
* dynamic position bias(DPB) use a MLP to generate the so-called relative position embedding to address the issue of variable token sizes  

[Transformer_architecture_positional_encoding_explanation](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
* absolute position encoding explanation  

[Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf)
* relative implementation explanation [link](https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a)  

[Rethinking and Improving Relative Position Encoding for Vision Transformer](https://arxiv.org/pdf/2107.14222.pdf)
* investigate different four relative position encodings for vision tasks under two modes: bias and contextual
