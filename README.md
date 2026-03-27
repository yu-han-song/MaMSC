#  Mamba-based Multi-Scale Classiﬁcation Framework for sEMG Gesture Recognition 

</div>

## Contributions

* We propose a novel multi-scale classification framework based on multipath Mamba for gesture recognition using sEMG signals. This framework achieves decoupling and fusion of multi-scale time-frequency features through a hybrid frequency-temporal decomposition and inverse reconstruction module.
* We design a multipath stacked Mamba encoder to perform dedicated intra-scale modeling for high-frequency, low-frequency, and time-domain residual signals, enhancing discriminative sequence representation capabilities.
* Extensive experiments on the Ninapro DB2 dataset demonstrate that this method performs excellently on intra-subject gesture recognition tasks. With only a minimal amount of target user data for fine-tuning across subject tasks, the model achieves 93.56\% accuracy, exhibiting low calibration costs and strong potential for real-world interaction adaptation.

## Architecture

![image-20260327202332092](D:\roaming\Typora\typora-user-images\image-20260327202332092.png)

## Full Results

![image-20260327202416705](D:\roaming\Typora\typora-user-images\image-20260327202416705.png)



The source code associated with this paper is currently under embargo and will be released after the paper is officially published.
