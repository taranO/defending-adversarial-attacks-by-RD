# CVPR ID 5842 

## Python implementation of "Defending against adversarial attacks by randomized diversification"

A  randomized diversification is a defense strategy against the adversarial attacks in a gray-box scenario. The gray-box attacks assume that the architecture of the classifier, the defense mechanism and the training data set are known to the attacker. The attacker does not only have an access to a secret key and to the internal states of the system at the test time. 

### Train

The multi-channel architecture with a randomized diversification can be trained for any type of deep classifiers and suits for any training data.

#### Reqirements
* keras
* numpy
* scipy.fftpack

### Test

For the test the adversarial examples were generate by using the attack propodsed by 
> Carlini Nicholas and Wagner David:  
> [Towards evaluating the robustness of neural networks](https://arxiv.org/pdf/1608.04644.pdf) 

The python attacks implementation was taken from https://github.com/carlini/nn_robust_attacks

The attcked data are available at https://drive.google.com/file/d/1__1XOZN8zDIfm0O4HExavx9eUoPwMKmQ/view?usp=sharing






