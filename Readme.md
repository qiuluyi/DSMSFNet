### ISPRS Vaihingen & ISPRS Potsdam dataset
##### Link: https://pan.baidu.com/s/1EWW9w6BVNpEg-CJXhrM-bg
##### password: umgt

### step 1 -> python cut_data.py
 In order to improve the generalization ability of the model and avoid overfitting of the model during the training process. In the training phase, we use some data enhancement operations to increase the amount of data.
 
### step 2 -> Setting parameters in config.py
Moreover, due to the server memory limitation, the high-resolution remote sensing images cannot be directly input into the proposed DCNN for training. 
Therefore, we first slide the IRRG image and DSM auxiliary data in the ISPRS dataset into 512×512 size data, and set the overlap stride to 32px.
Then, We train all our models using the Adam optimization algorithm with a base learning rate of 0.001, a momentum of 0.9, a weight decay of 0.99 and a batch size of 4. 

### step 3 -> python train.py
The root of the DSMSFNet network used is at ../networks/deeplab/DSMSFNet.py.
Users can adjust the network architecture according to their own needs.

### step 4 -> cd ./tools and python metrics.py
In order to be able to effectively verify the effectiveness of the proposed method, we use the classical evaluation metrics to evaluate the network model proposed in this model, including: F1 score and Overall Accuracy.


Related comparison methods
|Method|Date|Author|Title|Source|Code|
|---|---|---|---|---|---|
|SVL_3|2014|Gerke M.|Use of the stair vision library within the ISPRS 2D semantic labeling benchmark|[Arxiv](http://tailieuso.tlu.edu.vn/handle/DHTL/5106)|None|
|UT_Mev|2015|Speldekamp T, Fries C, Gevaert C, et al.|Automatic semantic labelling of urban areas using a rule-based apprOverall Accuracy ch and realized with MeVisLab|[Technical Report](https://webapps.itc.utwente.nl/librarywww/papers_2015/general/gerke_aut.pdf)|None|
|DST_2|2016|Sherrah J.|Fully convolutional networks for dense semantic labelling of high-resolution aerial imagery|[Online abstracts of the ISPRS benchmark on urban object classification and 3D building reconstruction](https://arxiv.org/pdf/1606.02585.pdf)|None|
|GSN3|2017|Wang H, Wang Y, Zhang Q, et al.|Gated convolutional neural network for semantic segmentation in high-resolution images|[Remote Sensing](https://www.mdpi.com/2072-4292/9/5/446/htm)|None|
|KLab_2|2018|Kemker R, Salvaggio C, Kanan C.|Algorithms for semantic segmentation of multispectral remote sensing imagery using deep learning|[ISPRS journal of photogrammetry and remote sensing](https://pdf.sciencedirectassets.com/271826/1-s2.0-S0924271618X00112/1-s2.0-S0924271618301229/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIBoZCIoqzn%2BkTMpavGEI335nAkOTtupPXtw0Dl91%2B7WTAiBxi6Evz1gOxkhLFj8EPKy089BMRPwq63hAf6WUeeDTNCr6AwgnEAQaDDA1OTAwMzU0Njg2NSIMgKjldk1sEb8JpYiAKtcDEF1woCUCATq8Dx%2BpTm3cPW%2By%2FSfA5xmbCLzEngErcfeW3iknbab2VSzpLUrPGKU7fcKPlT5MWBIar1X5loWolXVGqqtY50Ed11cOw64sw3hgM0fVPtf42ZUkO%2BbOIa8p1cPN9uZ9i1O9nl9t67caXimz8BB%2FX%2FVqUALyLAU10osww2suEkyfqVquhSQ7RFMlwSRGtBg7Feq3QCKV28xuoQDwuGV9bWT5lU%2FLxySgfEnESL5FgiLrqctelJpHU3lwmul1dzbPMuwsRLwJOJLJ1GY%2Fyk3FNvc5Es7axeJin0QQF40TTO7dwi1qc9C%2BQthBAECnN1mlpsq9hUCuuAVJ%2FAgcXyqBBKgfP5EGCKhpZAOSQzgOji7drONHcyjU2H1Xeccq9qavB%2FPo0bdqKDLPpTCfntCx7SIivmXqasB9RlnWbEPpH58NqaNQMfWIkOfr6n%2FDHbrdNSG3C2LEqsSxqJavYSGvQqsTjqvBBnDnqPjRp6RTvmrAVLPhBGdKhwaNVtZb2sb2PMrdmKxUHp6L6hxlm0QH4RlTDsEllKuu5%2F2AQuXK%2BtU02%2FZ57%2B%2Fxw8QE0WO7YsYvbQ6wzGTnY4xshXA%2BLgITpsfJrnWijamt9KDC%2BdpJn2ynMPanmpIGOqYBsLo2KsxAQmQG44CLvP6eWdX0q1qS55fvEWDkqjL8OwS1moIQOAnN%2Bb3B7tJvYniQ70chKsC%2BXnPkFiK4d8G34oRkFtbHyxWh1Bwfara1ffAc9Rznn7DcHgYOUPS4bXlfYyTzQcJv2A%2FiZhdijHWirTT8%2B6foJLLl9Azh5SLJvXaX13XVpr2q99vNM5BByVDvmisyPwdCQjGxHRSshooPisfMCc522g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220401T064754Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5TLFX2O6%2F20220401%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=7336b8a1582c190916dab169d3f228c82f038428ba6ae82eb6c379492b31eee4&hash=c81a5aff5595a3ba28f73b8dede69dbde1b55968d7ed8c234857e2d3dd4ed00b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0924271618301229&tid=spdf-fb40f2c9-544e-4b8b-beef-56b4b68643b9&sid=7892c08c7b057646487a93f3f41e2c82452egxrqa&type=client&ua=53075005575f525554&rr=6f4f60447f393ce2)|[Code](https://github.com/rmkemker/RIT-18)
|CVEO|2018|Chen G, Zhang X, Wang Q, et al.|Symmetrical dense-shortcut deep fully convolutional networks for semantic segmentation of very-high-resolution remote sensing images|[IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing](https://ieeexplore.ieee.org/abstract/document/8326706)|None|
|CASDE2|2018|Pan X, Gao L, Marinoni A, et al.|Semantic labeling of high resolution aerial imagery and LiDAR data with fine segmentation network|[Remote Sensing](https://www.mdpi.com/2072-4292/10/5/743/htm)|None|
|CONC_4|2018|Forbes T, He Y, Mudur S, et al.|Aggregated residual convolutional neural network for multi-label pixel wise classification of geospatial features|[Online abstracts of the ISPRS benchmark on urban object classification and 3D building reconstruction](https://www.isprs.org/education/benchmarks/UrbanSemLab/results/papers/abstract_CONC1.pdf)|None|
|ONE_7|2018|Audebert N, Le Saux B, Lefèvre S.|Beyond RGB: Very high resolution urban remote sensing with multimodal deep networks|[ISPRS Journal of Photogrammetry and Remote Sensing](https://www.sciencedirect.com/science/article/pii/S0924271617301818)|None|
|RIT_7|2018|Piramanayagam S, Saber E, Schwartzkopf W, et al.|Supervised classification of multisensor remotely sensed images using a deep learning framework|[Remote sensing](https://www.mdpi.com/2072-4292/10/9/1429/htm)|None|
|DLR_9|2018|Marmanis D, Schindler K, Wegner J D, et al.|Classification with an edge: Improving semantic image segmentation with boundary detection|[ISPRS Journal of Photogrammetry and Remote Sensing](https://www.sciencedirect.com/science/article/pii/S092427161630572X)|[Code](https://github.com/deep-unlearn/ISPRS-Classification-With-an-Edge)|
|CASIA2|2018|Liu Y, Fan B, Wang L, et al.|Semantic labeling in very high resolution images via a self-cascaded convolutional neural network|[ISPRS Journal of Photogrammetry and Remote Sensing](https://www.sciencedirect.com/science/article/pii/S0924271617303854)|[Code](https://github.com/Yochengliu/ScasNet)|
|BUCTY5|2019|Yue K, Yang L, Li R, et al.|TreeUNet: Adaptive tree convolutional neural networks for subdecimeter aerial image segmentation|[ISPRS Journal of Photogrammetry and Remote Sensing](https://www.sciencedirect.com/science/article/pii/S0924271619301741)|[Code]()|
|UFMG_4|2019|Nogueira K, Dalla Mura M, Chanussot J, et al.|Dynamic multicontext segmentation of remote sensing images based on convolutional networks|[IEEE Transactions on Geoscience and Remote Sensing](https://ieeexplore.ieee.org/abstract/document/8727958)|[Code](https://github.com/keillernogueira/dynamic-rs-segmentation/)|
|HUSTW3|2019|Sun Y, Tian Y, Xu Y.|Problems of encoder-decoder frameworks for high-resolution remote sensing image segmentation: Structural stereotype and insufficient learning|[Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231218313821)|None|
|LANet|2020|Ding L, Tang H, Bruzzone L.|LANet: Local attention embedding to improve the semantic segmentation of remote sensing images|[IEEE Transactions on Geoscience and Remote Sensing](https://ieeexplore.ieee.org/abstract/document/9102424)|None|
|CGFDN|2020|Zhou F, Hang R, Liu Q.|Class-guided feature decoupling network for airborne image segmentation|[IEEE Transactions on Geoscience and Remote Sensing](https://ieeexplore.ieee.org/abstract/document/9141325/)|None|
|BAM-UNet-sc|2021|Nong Z, Su X, Liu Y, et al.|Boundary-Aware Dual-Stream Network for VHR Remote Sensing Images Semantic Segmentation|[IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing](https://ieeexplore.ieee.org/abstract/document/9416898/)|None|
|MACANet|2021|Li X, Lei L, Kuang G.|Multilevel Adaptive-Scale Context Aggregating Network for Semantic Segmentation in High-Resolution Remote Sensing Images|[IEEE Geoscience and Remote Sensing Letters](https://ieeexplore.ieee.org/document/9470922)|[Code](https://github.com/RSIP-NUDT/MACANet)|
|SAGNN|2022|Diao Q, Dai Y, Zhang C, et al.|Superpixel-based attention graph neural network for semantic segmentation in aerial images|[Remote Sensing](https://www.mdpi.com/2072-4292/14/2/305/htm)|None|


### For more details, please contact the author: 458831254@qq.com
