## Defending against adversarial attacks by randomized diversification

The research was supported by the [SNF](http://www.snf.ch) project No. 200021_182063. 
##

PyTorch implementation of ["Defending against adversarial attacks by randomized diversification"](http://sip.unige.ch/projects/snf-it-dis/publications/cvpr-2019) 

A  randomized diversification is a defense strategy against the adversarial attacks in a gray-box scenario. The gray-box attacks assume that the architecture of the classifier, the defense mechanism and the training data set are known to the attacker. The attacker does not only have an access to a secret key and to the internal states of the system at the test time. 

<p align="center">
<img src="http://sip.unige.ch/files/3615/5264/5259/2019_CVPR_main_schema.png" height="350px" align="center">
<br/>
<br/>
Fig.1: Generalized diagram of the proposed multi-channel classifier.  
</p>

A multi-channel classifier forms the core of the proposed architecture and consists of four main building blocks:

* Pre-processing of the input data in a transform domain via a mapping <img vertical-align="middle;" src="http://sip.unige.ch/files/1415/5264/6029/2019_cvpr_001.png" alt="2019_cvpr_001.png" height="25">
* Data independent processing <img style="vertical-align: middle;" src="http://sip.unige.ch/files/2315/5264/6226/2019_cvpr_002.png" alt="2019_cvpr_002.png" height="25"> serves as a defense against gradient back propagation to the direct domain.
* Classification block can be represented by any family of classifiers.
* Aggregation block can be represented by any operation ranging from a simple summation to learnable operators adapted to the data or to a particular adversarial attack.

The chain of the first 3 blocks can be organized in a parallel multi-channel structure that is followed by one or several aggregation blocks. The final decision about the class is made based on the aggregated result. The rejection option can be also naturally envisioned.



## Reqirements
* keras
* numpy
* scipy.fftpack

## Train

The multi-channel architecture with a randomized diversification can be trained for any type of deep classifiers and suits for any training data.
  
    $ python train_model_multi_channel.py --type=mnist --epochs=1000

## Test

For the test the adversarial examples were generate by using the attack propodsed by 
> Carlini Nicholas and Wagner David:  
> [Towards evaluating the robustness of neural networks](https://arxiv.org/pdf/1608.04644.pdf) 

The python attacks implementation was taken from https://github.com/carlini/nn_robust_attacks

The attcked data are available at https://cuicloud.unige.ch/index.php/s/QcSPGSLSRCzc2gm
  
    $ python test_model_multi_channel.py --type=mnist --attack_type=carlini_l2 --data_dir=data/attacked --epoch=1000

## Citation
O. Taran, S. Rezaeifar, T. Holotyak, and S. Voloshynovskiy, "Defending against adversarial attacks by randomized diversification," in Proc. The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, USA, 2019. 
  
    @inproceedings { Taran2019cvpr,
      author = { Taran, Olga and Rezaeifar, Shideh and Holotyak, Taras and Voloshynovskiy, Slava },
      booktitle = { The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) },
      title = { Defending against adversarial attacks by randomized diversification },
      address = { Long Beach, USA },
      month = { June },
      year = { 2019 }
    }




