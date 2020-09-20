# Survey-of-Intelligent-algorithms-for-unusual-weather-in-Internet-of-Things

## 1. Derain algorithms'Code in paper

| Algorithms | Paper Link | Projects Link |  
| ------------------- | ------------------------------------------------------- | ------------------- |
|   DDN    | [Paper Link](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf)   |[Code Link](https://github.com/XMU-smartdsp/Removing_Rain)                 |  
| DerainNet    | [Paper Link](https://arxiv.org/abs/1609.02087 )    | [Code Link](https://xueyangfu.github.io/projects/tip2017.html)    | 
| LPNet    | [Paper Link](https://arxiv.org/abs/1805.06173)    | [Code Link](https://xueyangfu.github.io/projects/LPNet.html)    | 
| PReNet    | [Paper Link](https://arxiv.org/abs/1901.09221)    | [Code Link](https://github.com/csdwren/PReNet)    | 
| GP-based SSL    | [Paper Link](https://arxiv.org/abs/2006.05580)    | [Code Link](https://github.com/rajeevyasarla/Syn2Real)    | 


### Testing execution time
$T_{total} = T_{load \ model} + n * ( T_{read} + T_{process} + T_{write} )$ , where $n$ is testing image amount.


[How to divide execution time (code)](https://www.notion.so/How-to-divide-execution-time-code-9538f9190c06425ba5be58c327b27a7a)


## 2. Detection Code and Model


| Algorithms | Code Link | Pretrain Model Link |  
| ------------------- | ------------------------------------------------------- | ------------------- |
|   Faster-RCNN       | [Code Link](https://github.com/rbgirshick/py-faster-rcnn)     |  sh./data/scripts/fetch_faster_rcnn_models.sh   |  
|   RetinaNet         | [Code Link](https://github.com/fizyr/keras-retinanet)   |  MSCOCO pretrain model    |  
|   YoloV3            | [Code Link](https://github.com/pjreddie/darknet)    | https://pjreddie.com/media/files/yolov3.weights   |   
|   SSD-512           | [Code Link](https://github.com/FreeApe/VGG-or-MobileNet-SSD)    | https://drive.google.com/file/d/0BzKzrI_SkD1_NVVNdWdYNEh1WTA/view  | 

## 3. Synthetic Data and Real Data 

| Data Type | Download Link |  Download Link |  
| ------------------- | ------------------------------------------------------- | ------------------- |
| Rain Mist        | [Real](https://pan.baidu.com/s/1lB5jQgGr-5aGuxT5Z8_YmA) Access Code: 6h55  | [Synthetic](https://pan.baidu.com/s/1JYtoefuCHovSE2emXP6LwA) Access Code: 8kae     |  
| Rain Drop        | [Real](https://pan.baidu.com/s/1TlDY2XV2U3Et2egRO96t_g) Access Code: n6xf  | [Synthetic](https://pan.baidu.com/s/1qFrtVvPLqc1FnsmHlXYgiA) Access Code : wscw     |   
| Rain Streak      | [Real](https://pan.baidu.com/s/1XctM1xT9KKq3JU_OXPJiLg) Access Code: npsy  | [Synthetic](https://pan.baidu.com/s/11t4XIx6f3CEvmOw2XO9fqQ) Access Code: drxn     | 
|  the dataset of Deep-Network      |  [Github](https://github.com/jinnovation/rainy-image-dataset) |
|Rain100H, Rain100L, Rain1400 and Rain12       |  [Onedrive](https://onedrive.live.com/?authkey=%21AIYIy8ZKL9kkmd4&id=66CE859AB42DFA2%2130078&cid=066CE859AB42DFA2) | 
|Rain12600, RainTrainL, RainTrainH | [Onedrive](https://onedrive.live.com/?authkey=%21AIYIy8ZKL9kkmd4&id=66CE859AB42DFA2%2130078&cid=066CE859AB42DFA2) | |

**We note that*:

**RainTrainL/Rain100L** and **RainTrainH/Rain100H** are synthesized by [Yang Wenhan](https://github.com/flyywh).
*  RainTrainH has 1800 rainy images for training, and Rain100H has 100 rainy images for testing.
**Rain12600/Rain1400** is from [Fu Xueyang](https://xueyangfu.github.io/) and **Rain12** is from [Li Yu](http://yu-li.github.io/).*
* Rain12600 and Rain1400 contains 1,000 clean images. Each clean image was used to generate 14 rainy images with different streak orientations and magnitudes.Rain12600 has 900 clean images for "training" and Rain1400 has 100 clean images for "testing".
* Rain12  only includes 12 rainy images

##  4. Image Quality Metrics
* PSNR (Peak Signal-to-Noise Ratio) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4550695) [[Matlab Code]](https://www.mathworks.com/help/images/ref/psnr.html) [[Python Code]](https://github.com/aizvorski/video-quality)
* SSIM (Structural Similarity) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1284395) [[Matlab Code]](http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m) [[Python Code]](https://github.com/aizvorski/video-quality/blob/master/ssim.py)
* VIF (Visual Quality) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1576816) [[Matlab Code]](http://sse.tongji.edu.cn/linzhang/IQA/Evalution_VIF/eva-VIF.htm)
* FSIM (Feature Similarity) [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5705575) [[Matlab Code]](http://sse.tongji.edu.cn/linzhang/IQA/FSIM/FSIM.htm))




## 5. Some related algorithms and Paper Link

## The rain models
* Automatic single-image-based rain streaks removal via image decomposition (TIP2012), Kang et al [[PDF]](http://www.ee.nthu.edu.tw/cwlin/Rain_Removal/tip_rain_removal_2011.pdf) [[Code]](http://www.ee.nthu.edu.tw/cwlin/pub/rain_tip2012_code.rar)

* Removing rain from a single image via discriminative sparse coding [[PDF]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Luo_Removing_Rain_From_ICCV_2015_paper.pdf)

* Depth-attentional Features for Single-image Rain Removal [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Depth-Attentional_Features_for_Single-Image_Rain_Removal_CVPR_2019_paper.pdf)
* Frame-Consistent Recurrent Video Deraining with Dual-Level Flow [[PDF]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Frame-Consistent_Recurrent_Video_Deraining_With_Dual-Level_Flow_CVPR_2019_paper.pdf)

## model-driven
  * Guided image filtering (ECCV2010), He et al. [[Project]](http://kaiminghe.com/eccv10/index.html) [[PDF]](http://kaiminghe.com/publications/eccv10guidedfilter.pdf) [[Code]](http://kaiminghe.com/eccv10/guided-filter-code-v1.rar)
  * Removing rain and snow in a single image using guided filter (CSAE2012), Xu et al. [[PDF]](https://ieeexplore_ieee.gg363.site/abstract/document/6272780)
  * An improved guidance image based method to remove rain and snow in a single image (CIS2012), Xu et al. [[PDF]](https://pdfs.semanticscholar.org/6eac/36e3334dd0c9188b5a61af73909dcbfff39c.pdf)
  * Single-image deraining using an adaptive nonlocal means filter (ICIP2013), Kim et al. [[PDF]](https://ieeexplore_ieee.gg363.site/abstract/document/6738189)
  * Single-image-based rain and snow removal using multi-guided filter (NIPS2013), Zheng et al. [[PDF]](https://pdfs.semanticscholar.org/f111/54e4e1adbde9f24b25fd2d98337a759d8b21.pdf)
  * Single image rain and snow removal via guided L0 smoothing filter (Multimedia Tools and Application2016), Ding et al. [[PDF]](https://link_springer.gg363.site/article/10.1007/s11042-015-2657-7)
  

  * Automatic single-image-based rain streaks removal via image decomposition (TIP2012), Kang et al [[PDF]](http://www.ee.nthu.edu.tw/cwlin/Rain_Removal/tip_rain_removal_2011.pdf) [[Code]](http://www.ee.nthu.edu.tw/cwlin/pub/rain_tip2012_code.rar)
  * Self-learning-based rain streak removal for image/video (ISCS2012), Kang et al. [[PDF]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.701.3957&rep=rep1&type=pdf)
  * Single-frame-based rain removal via image decomposition (ICA2013), Fu et al. [[PDF]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.707.1053&rep=rep1&type=pdf)
  * Exploiting image structural similarity for single image rain removal (ICIP2014), Sun et al.  [[PDF]](http://mml.citi.sinica.edu.tw/papers/ICIP_2014_Sun.pdf)
  * Visual depth guided color image rain streaks removal using sparse coding (TCSVT2014), Chen et al [[PDF]](https://ieeexplore.ieee.org/document/6748866/)
  * Removing rain from a single image via discriminative sparse coding (ICCV2015), Luo et al [[PDF]](http://ieeexplore.ieee.org/document/7410745/) [[Code]](http://www.math.nus.edu.sg/~matjh/download/image_deraining/rain_removal_v.1.1.zip)
  * Rain streak removal using layer priors (CVPR2016), Li et al [[PDF]](https://ieeexplore.ieee.org/document/7780668/) [[Code]](http://yu-li.github.io/)
  * Single image rain streak decomposition using layer priors (TIP2017), Li et al [[PDF]](https://ieeexplore.ieee.org/document/7934436/)
  * Error-optimized dparse representation for single image rain removal (IEEE TIE2017), Chen et al [[PDF]](https://ieeexplore.ieee.org/abstract/document/7878618/)
  * A hierarchical approach for rain or snow removing in a single color image (TIP2017), Wang et al. [[PDF]](http://ieeexplore.ieee.org/abstract/document/7934435/)
  * Joint bi-layer optimization for single-image rain streak removal (ICCV2017), Zhu et al. [[PDF]](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Joint_Bi-Layer_Optimization_ICCV_2017_paper.html)
  * Convolutional sparse and low-rank codingbased rain streak removal (WCACV2017), Zhang et al [[PDF]](https://ieeexplore_ieee.gg363.site/abstract/document/7926728/)
  * Joint convolutional analysis and synthesis sparse representation for single image layer separation (CVPR2017), Gu et al [[PDF]](http://openaccess.thecvf.com/content_iccv_2017/html/Gu_Joint_Convolutional_Analysis_ICCV_2017_paper.html) [[Code]](https://sites.google.com/site/shuhanggu/home)
  * Single image deraining via decorrelating the rain streaks and background scene in gradient domain (PR2018)， Du et al [[PDF]](https://www.sciencedirect.com/science/article/pii/S0031320318300700)
  
## data-driven
  * Restoring an image taken through a window covered with dirt or rain (ICCV2013), Eigen et al. [[Project]](https://cs.nyu.edu/~deigen/rain/) [[PDF]](http://openaccess.thecvf.com/content_iccv_2013/papers/Eigen_Restoring_an_Image_2013_ICCV_paper.pdf) [[Code]](https://cs.nyu.edu/~deigen/rain/restore-dirt-rain.tgz)
  * Attentive generative adversarial network for raindrop removal from a single image (CVPR2018), Qian et al [[Project]](https://rui1996.github.io/raindrop/raindrop_removal.html) [[PDF]](https://arxiv.org/abs/1711.10098)
  * Clearing the skies: A deep network architecture for single-image rain streaks removal (TIP2017), Fu et al. [[Project]](https://xueyangfu.github.io/projects/tip2017.html) [[PDF]](https://ieeexplore.ieee.org/abstract/document/7893758/) [[Code]](https://xueyangfu.github.io/projects/tip2017.html)
  * Removing rain from single images via a deep detail network (CVPR2017), Fu et al. [[Project]](https://xueyangfu.github.io/projects/cvpr2017.html) [[PDF]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf) [[Code]](https://xueyangfu.github.io/projects/cvpr2017.html)
  * Image de-raining using a conditional generative adversarial network (Arxiv2017), Zhang et al [[PDF]](https://arxiv.org/abs/1701.05957) [[Code]](https://github.com/hezhangsprinter/ID-CGAN)
  * Deep joint rain detection and removal from a single image (CVPR2017), Yang et al.[[Project]](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html) [[PDF]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_Deep_Joint_Rain_CVPR_2017_paper.pdf) [[Code]](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)
  * Residual guide feature fusion network for single image deraining (ACMMM2018), Fan et al. [[Project]](https://zhiwenfan.github.io/) [[PDF]](http://export.arxiv.org/pdf/1804.07493)
  * Fast single image rain removal via a deep decomposition-composition network (Arxiv2018), Li et al [[Project]](https://sites.google.com/view/xjguo/rain)) [[PDF]](https://arxiv.org/abs/1804.02688) [[Code]](https://drive.google.com/open?id=1TPu9RX7Q9dAAn5M1ECNbqtRDa9c6_WOt)
  * Density-aware single image de-raining using a multi-stream dense network (CVPR2018), Zhang et al [[PDF]](https://arxiv.org/abs/1802.07412) [[Code]](https://github.com/hezhangsprinter/DID-MDN)
  * Recurrent squeeze-and-excitation context aggregation net for single image deraining (ECCV2018), Li et al. [[PDF]](https://export.arxiv.org/pdf/1807.05698) [[Code]](https://github.com/XiaLiPKU/RESCAN)
  * Rain streak removal for single image via kernel guided cnn (Arxiv2018), Wang et al [[PDF]](https://arxiv.org/pdf/1808.08545.pdf)
  * Physics-based generative adversarial models for image restoration and beyond (Arxiv2018), Pan et al [[PDF]](https://arxiv.org/pdf/1808.00605.pdf)
  * Learning dual convolutional neural networks for low-level vision (CVPR2018), Pan et al [[Project]](https://sites.google.com/site/jspanhomepage/dualcnn) [[PDF]](https://arxiv.org/pdf/1805.05020.pdf) [[Code]](https://sites.google.com/site/jspanhomepage/dualcnn)
  * Non-locally enhanced encoder-decoder network for single image de-raining (ACMMM2018), Li et al [[PDF]](https://arxiv.org/pdf/1808.01491.pdf) [[Code]](https://github.com/AlexHex7/NLEDN)
  *  Unsupervised single image deraining with self-supervised constraints (ICIP2019), Jin et al [[PDF]](https://arxiv.org/pdf/1811.08575)
  * Progressive image deraining networks: A better and simpler baseline (CVPR2019), Ren et al [[PDF]](https://csdwren.github.io/papers/PReNet_cvpr_camera.pdf) [[Code]](https://github.com/csdwren/PReNet)
  * Spatial attentive single-image deraining with a high quality real rain dataset (CVPR2019), Wang et al [[Project]](https://stevewongv.github.io/derain-project.html) [[PDF]](https://arxiv.org/abs/1904.01538) [[Code]](https://github.com/stevewongv/SPANet)
  * Lightweight pyramid networks for image deraining (TNNLS2019), Fu et al [[PDF]](https://arxiv.org/pdf/1805.06173.pdf) [[Code]](https://xueyangfu.github.io/projects/LPNet.html)
  *  Joint rain detection and removal from a single image with contextualized deep networks (TPAMI2019), Yang et al [[PDF]](https://ieeexplore.ieee.org/document/8627954) [[Code]](https://github.com/flyywh/JORDER-E-Deep-Image-Deraining-TPAMI-2019-Journal)


