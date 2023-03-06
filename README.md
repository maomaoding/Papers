# Papers 

## Fundamental related
[On Calibration of Modern Neural Networks](https://arxiv.org/pdf/1706.04599.pdf)  
[Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf)
* when the minibatch size if multiplied by k, multiply the learning rate by k
* gradual warmup to ramp up the learning rate from a small to a large rate(allowing healthy convergence at the start of training)  
[Large Batch Training of convolutional networks](https://arxiv.org/pdf/1708.03888.pdf)
[ON THE VARIANCE OF THE ADAPTIVE LEARNING RATE AND BEYOND](https://arxiv.org/pdf/1908.03265.pdf)
* identify the variance issue of the adaptive learning rate and present a theoretical justification for the warmup heuristic, convergence issue is due to the undesirably large variance of the adaptive learning rate in the early stage of model training
* propose a new variant of Adam(RAdam)
[Truly shift-invariant convolutional neural networks](https://arxiv.org/pdf/2011.14214.pdf)
[Fourier Transform](https://www.princeton.edu/~cuff/ele201/kulkarni_text/frequency.pdf)
[MIND THE PAD – CNNS CAN DEVELOP BLIND SPOTS](https://arxiv.org/pdf/2010.02178.pdf)  
[An Analysis of Scale Invariance in Object Detection – SNIP](https://arxiv.org/pdf/1711.08189.pdf)

## Low-shot Learning
[Low-Shot Learning from Imaginary Data](https://arxiv.org/pdf/1801.05401.pdf)

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
[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934.pdf)
* [explanation blog](https://www.keep-current.dev/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis/)

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

[A Survey on Vision Transformer](https://arxiv.org/pdf/2012.12556.pdf)  
[Attention is not all you need: pure attention loses rank doubly exponentially with depth](https://arxiv.org/pdf/2103.03404.pdf)

## Lane Curve Parameter Detection  
[Structured Bird’s-Eye-View Traffic Scene Understanding from Onboard Images](https://arxiv.org/pdf/2110.01997.pdf)  

## BEV Segmentation & Detection
[NEAT: Neural Attention Fields for End-to-End Autonomous Driving](https://arxiv.org/pdf/2109.04456.pdf)
* Inspired by implicit shape representations, NEAT represents large dynamic scenes with a fixed memory footprint using a multi-layer perceptron query function  
[Gen-LaneNet: A Generalized and Scalable Approach for 3D Lane Detection](https://arxiv.org/pdf/2003.10656.pdf)  
[Learning to Predict 3D Lane Shape and Camera Pose from a Single Image via Geometry Constraints](https://arxiv.org/pdf/2112.15351.pdf)  
[Monocular 3D Object Detection: An Extrinsic Parameter Free Approach](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Monocular_3D_Object_Detection_An_Extrinsic_Parameter_Free_Approach_CVPR_2021_paper.pdf)  

## Monocular 3D Detection
[EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation](https://arxiv.org/pdf/2203.13254.pdf)

## Diffusion Model
[diffusion model](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process)