
<!-- TOC -->

- [Networks](#networks)
    - [Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](#deep-generative-image-models-using-a-laplacian-pyramid-of-adversarial-networks)
        - [Main idea](#main-idea)
        - [Laplacian Pyramid](#laplacian-pyramid)
        - [LAPGAN](#lapgan)
        - [Thinking](#thinking)
    - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial](#unsupervised-representation-learning-with-deep-convolutional-generative-adversarial)
        - [Main idea](#main-idea)
        - [Architecture](#architecture)
        - [Experiments](#experiments)
    - [Improved Techniques for Training GANs](#improved-techniques-for-training-gans)
        - [Main idea](#main-idea)
        - [Feature Matching](#feature-matching)
        - [Minibatch Discrimination](#minibatch-discrimination)
        - [Historical averaging](#historical-averaging)
        - [One-sided label smoothing](#one-sided-label-smoothing)
        - [Virtual batch normalization](#virtual-batch-normalization)
        - [Assessment of image quality](#assessment-of-image-quality)
        - [Semi-supervised learning](#semi-supervised-learning)
    - [Generative Image Modeling using Style and Structure Adversarial Networks](#generative-image-modeling-using-style-and-structure-adversarial-networks)
        - [Main idea](#main-idea)
        - [Structure-GAN](#structure-gan)
        - [Style-GAN](#style-gan)
        - [Style-GAN with Pixel-wise Constraints](#style-gan-with-pixel-wise-constraints)
        - [Joint Learning for S2-GAN](#joint-learning-for-s2-gan)
        - [Experiments](#experiments)
    - [Learning from Simulated and Unsupervised Images through Adversarial Training](#learning-from-simulated-and-unsupervised-images-through-adversarial-training)
        - [Main idea](#main-idea)
        - [Architecture](#architecture)
        - [Adversarial Loss with Self-Regularization](#adversarial-loss-with-self-regularization)
        - [Local adversarial loss](#local-adversarial-loss)
        - [perceptual loss](#perceptual-loss)
        - [Loss function](#loss-function)
    - [Face Synthesis from Visual Attributes via Sketch using Conditional VAEs and GANs](#face-synthesis-from-visual-attributes-via-sketch-using-conditional-vaes-and-gans)
        - [Main idea](#main-idea)
        - [Motivation](#motivation)
        - [Architecture](#architecture)
        - [Stage 1: Attribute-to-Sketch](#stage-1-attribute-to-sketch)
        - [Stage 2: Sketch-to-Sketch (S2S)](#stage-2-sketch-to-sketch-s2s)
        - [Stage 3: Sketch-to-Face (S2F)](#stage-3-sketch-to-face-s2f)
        - [Experiments](#experiments)
    - [Progressive Growing of GANs for Improved Quality, Stability, and Variation](#progressive-growing-of-gans-for-improved-quality-stability-and-variation)
        - [Main idea](#main-idea)
        - [Progressive Growing of GANs](#progressive-growing-of-gans)
        - [Increasing Variation](#increasing-variation)
        - [Normalization in Generator and Discriminator](#normalization-in-generator-and-discriminator)
        - [Multi-Scale Statistical Similarity for Assessing GAN Results](#multi-scale-statistical-similarity-for-assessing-gan-results)
        - [Experiments](#experiments)

<!-- /TOC -->

# Networks
## Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks

### Main idea
The authors introduce a generative parametric model capable of producing high quality samples of natural images, using a cascade of convolutional networks within Laplacian pyramid framework(LPGAN) to generate images in a coarse-to-fine fashion.

### Laplacian Pyramid
The Laplacian pyramid is a linear invertible image representation consisting of a set of band-pass images, spaced an octave apart, plus a low-frequency residual. I(0) = I and I(k) is k repeated applications of downsampling.

![](img/lpgan_lp.png)

Starting at the coarsest level, we repeatedly upsample and add the difference image h at the next finer level until we get back to the full resolution image.

![](img/lpgan_lp1.png)

### LAPGAN
In the reconstruction procedure, a set of conditional generative convet models {G0,…,Gk}, each of which captures the distribution of coefficients h_k for natural images at a different level of the Laplacian pyramid.

![](img/lpgan_theory1.png)

In the training procedure, G(k) is a convnet which uses a coarse scale version of the image l(k) = u(I(k+1)) as an input, as well as noise vector z(k). D(k) takes as input h(k) or h’(k), along with the low-pass image l(k), and predicts if the image was real or generated. 

![](img/lpgan_theory2.png)

### Thinking
(1) Breaking the generation into successive refinements is the key idea in this work.
(2) The authors never make any attempt to train a network to discriminate between the output of a cascade and a real image and instead focus on making each step plausible.
(3) The independent training of each pyramid level has the advantage that it is far more difficult for the model to memorize training examples – a hazard when high capacity deep networks are used.

## Unsupervised Representation Learning with Deep Convolutional Generative Adversarial

### Main idea
(1) The authors propose and evaluate a set of constraints on the architectural topology of convolutional GANs, named DCGAN, making them stable to train in most datasets.
(2) The authors visualize the filters learnt by GANs and empirically show that specific filters have learned to draw specific obejects.
(3) The authors show that the generators have interesting vector arithmetic properties allowing for easy manipulation of many semantic qualities of generated samples.

### Architecture
1. All convolutional net replaces deterministic spatial pooling functions (such as maxpooling).
Discriminator: strided convolutions
![](img/dcgan_conv.gif)
Generator: fractional strided convolutions
![](img/dcgan_dconv.gif)

2. Remove fully connected hidden layers for deeper architectures.
![](img/dcgan_nofc.png)

3. Use batchnorm in both the generator and the discriminator, which helps deal with training problems that arise due to poor initialization and helps gradient flow in deeper models

4. Use ReLU activation in generator for all layers except for the output which uses Tanh and Use LeakyReLU activation in the discriminator for all layers

### Experiments
(1) Validation of DCGAN capabilities using GANs as a feature extractor

(2) Walking in the latent space(z) results in semantic changes

(3) Using guided backpropagation to visualize the features learnt by discriminator

(4) Remove certain objects by dropping out relevant feature activations

(5) Perform vector arithmetic on the Z vectors to evaluate learned representation space

## Improved Techniques for Training GANs
### Main idea
The authors present a variety of new architectural features and training procedures for GANs framework, which help greatly in semi-supervised classification and the generation of images that humans find visually realistic. 

### Feature Matching
Instead of directly maximizing the output of the discriminator, the new objective requires the generator to generate data that matches the statistics of the real data, where we use the discriminator only to specify the statistics that we think are worth matching. Specifically, we train the generator to match the expected value of the features on an intermediate layer of the discriminator. 

![](img/imgan_fm.png)

### Minibatch Discrimination
Discriminator model that looks at multiple examples in combination, rather than in isolation, could potentially help avoid collapse of the generator. Modelling the closeness between examples in a minibatch is as follows:

![](img/imgan_md.png)

### Historical averaging
modify each player’s cost to include a term:

![](img/imgan_ha.png)

where θ[i] is the value of the parameters at past time i. This approach was able to find equilibria of low-dimensional, continuous non-convex games and some specific cases.

### One-sided label smoothing
Replacing the 0 and 1 targets for a classifier with smoothed values, like 0.9 or 0.1, is shown to reduce the vulnerability of neural networks to adversarial examples. Replacing positive classification targets with α and negative targets with β:

![](img/imgan_ls.png)

smooth only the positive labels to α, leaving negative labels set to 0. 

### Virtual batch normalization
(1) Batch normalization causes the output of a neural network for an input example x to be highly dependent on several other inputs x’ in the same minibatch. 

(2) To avoid this problem we introduce virtual batch normalization (VBN), in which each example x is normalized based on the statistics collected on a reference batch of examples that are chosen once and fixed at the start of training, and on x itself. 

### Assessment of image quality
1. One intuitive metric of performance can be obtained by having human annotators judge the visual quality of samples, but the metric varies depending on the setup of the task and the motivation of the annotators

2. An automatic method to evaluate samples: images that contain meaningful objects should have a conditional label distribution p(y|x) with low entropy. Moreover, we expect the model to generate varied images, so the marginal ∫ p(y|x = G(z)) 𝑑𝑧  should have high entropy.
![](img/imgan_as.png)

### Semi-supervised learning
We can do semi-supervised learning with any standard classifier by simply adding samples from the GAN generator G to our data set, labeling them with a new “generated” class y = K + 1.
![](img/imgan_ss.png)


## Generative Image Modeling using Style and Structure Adversarial Networks
### Main idea
The authors factorize the image generation process and propose Style and Structure Generative Adversarial Network (S2-GAN) consisting of two components: 
- the Structure-GAN generates a surface normal map; 
- the Style-GAN takes the surface normal map as input and generates the 2D image. 

![](img/ssgan_arch.png)

### Structure-GAN
We train a Structure-GAN using the ground truth surface normal from Kinect. Because the perspective distortion of texture is more directly related to normals than to depth, we use surface normal to represent image structure in this paper. 

![](img/ssgan_stru.png)

### Style-GAN
Given the RGB images and surface normal maps from Kinect, we modify our generator network to a conditional GAN. The surface normal maps are given as additional inputs for both the generator G and the discriminator D, which enforces the generated image to match the surface normal map.

![](img/ssgan_style.png)

### Style-GAN with Pixel-wise Constraints
We make the following assumption: if the generated image is real enough, it can be used for reconstructing the surface normal maps. Thus, we propose to add a pixel-wise constraint to explicitly guide the generator to align the outputs with the input surface normal maps.

![](img/ssgan_style_.png)

We train Fully Convolutional Network(FCN) wieh classification for surface normal estimation, using the RGBD data which provides indoor scene images and ground truth surface normals.

![](img/ssgan_style_train.png)

### Joint Learning for S2-GAN
After training the Structure-GAN and Style-GAN independently, we merge all networks and train them jointly with removing the FCN constraint.

![](img/ssgan_jointly.png)

### Experiments

![](img/ssgan_exp.png)




## Learning from Simulated and Unsupervised Images through Adversarial Training
### Main idea
1. To reduce the gap between synthetic and real image distributions, the authors propose Simulated+Unsupervised (S+U) learning, where the task is to learn a model to improve the realism of a simulator’s output using unlabeled real data, while preserving the annotation information from the simulator.

2. The authors make several key modifications to the standard GAN algorithm to preserve annotations, avoid artifacts, and stabilize training: (i) a ‘self-regularization’ term, (ii) a local adversarial loss, and (iii) updating the discriminator using a history of refined images. 

### Architecture
The key requirement for S+U learning is that the refined image x’ should look like a real image in appearance while preserving the annotation information from the simulator.

![](img/Simgan_arch.png)

### Adversarial Loss with Self-Regularization
To add realism, we train our refiner network using an adversarial loss. The loss of discriminator network and refiner network are followed:

![](img/Simgan_adv.png)

where D(.) is the probability of the input being a synthetic image, and 1 − D(.) is that of a real one. 

To preserve annotations information of the simulator, we propose using a slef regularization loss that minimizes per-pixel difference between a feature transform of the synthetic and refined images.

![](img/Simgan_reg.png)

where ψ is the mapping from image space to a feature space, and |.|1 is the L1 norm. The feature transform can be an identity map, image derivatives, mean of color channels, or a learned transformation such as a convolutional neural network

The overall refiner loss function: 

![](img/Simgan_refiner.png)

### Local adversarial loss
Furthermore, to avoid drifting and introducing spurious artifacts while attempting to fool a single stronger discriminator, we limit the discriminator’s receptive field to local regions, resulting in multiple local adversarial losses per image.
- When we train a single strong discriminator network, the refiner network tends to over-emphasize certain image features to fool the current discriminator network, leading to drifting and producing artifacts.
- Any local patch sampled from the refined image should have similar statistics to a real image patch.
- This division not only limits the receptive field, and hence the capacity of the discriminator network, but also provides many samples per image for learning the discriminator network.

![](img/Simgan_local.png)

### perceptual loss
By minimizing the perceptual adversarial loss function LT with respect to parameters of T, we will encourage the network T to generate image T(x) that has similar high-level features with its ground-truth y. 

![](img/Simgan_perceptual.png)

If the sum of discrepancy between the current learned representations of transformed image T(x) and ground-truth image y is less than the positive margin m, the loss function LD will upgrade the discriminative network D for new high-dimensional spaces, which discover the discrepancy that still exist between the transformed images and corresponding ground-truth. 

### Loss function
Given a pair of data (x, y) ∈ (X(input), Y(ground-truth)), the loss function of image transformation network J(T) and the loss function of discriminative network J(D) are formally defined as:

![](img/Simgan_loss.png)


## Face Synthesis from Visual Attributes via Sketch using Conditional VAEs and GANs

### Main idea
the authors synthesize face images from attributes and
text descriptions in three stages: 
(1) Synthesis of facial sketch from attributes using a CVAE architecture, 
(2) Enhancement of coarse sketches to produce sharper sketches using a GANbased framework, and 
(3) Synthesis of face from sketch using another GAN-based network

### Motivation
Visual description-based face synthesis has many applications in law enforcement and entertainment.

### Architecture
Three-stage training network: 

![](img/FSASgan_train.png)

Testing phase:

![](img/FSASgan_test.png)

### Stage 1: Attribute-to-Sketch
In the A2S stage, the authors adapt the CVAE architecture. Given a texture attribute vector a, noise vector n, and ground-truth sketch s, we aim to learn a model Pθ(s|a z), which can model the distribution of s and generate s(r). The objective is to find the best parameter θ which maximizes the loglikelihood log Pθ(s|a).

![](img/FSASgan_A2S.png)

The encoder q(φ) takes sketch and attributes as input, whereas q(β) takes noise and attribute vectors as input. The overall loss function of the A2S stage is as follows: 

![](img/FSASgan_loss1.png)

KL(Qφ(z|s a)||Pθ(z)) and KL(Qβ(z|n a)||Pθ(z)), are the regularization terms in order to enforce the latent variable z ~ Qφ(z|s a) and z ~ Qβ(z|n a) both match the prior normal distribution, Pθ(z).

### Stage 2: Sketch-to-Sketch (S2S)
S2S network consists of a generator sub-network G2 (based on UNet and DenseNet architectures) conditioned on the encoded attribute vector from the A2S stage and a patch-based discriminator subnetwork D2. 

![](img/FSASgan_S2S.png)

UNet incorporates longer skip connections to preserve low-level features, DenseNet employs short range connections within microblocks resulting in maximum information flow between layers in addition to an efficient network.

Additionally, patch-based discriminator ensures preserving of high-frequency details which are usually lost when only L1 loss is used. 

![](img/FSASgan_loss2.png)

### Stage 3: Sketch-to-Face (S2F)
The visual attribute vector is combined with the latent representation to produce attribute-preserved image reconstructions.

![](img/FSASgan_S2F.png)

The network parameters for the S2F stage are learned by minimizing

![](img/FSASgan_loss3.png)

### Experiments
the authors show the image synthesis capability of our network by manipulating the input attribute and noise vectors. Note that, the testing phase of our network takes attribute vector and noise as inputs and produces face reconstruction as the output. 

![](img/FSASgan_exp.png)

![](img/FSASgan_exp2.png)

## Progressive Growing of GANs for Improved Quality, Stability, and Variation
### Main idea
1) The authors describe a new training methodology for GANs to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresss.
2) The authors propose a simple way to increase the variation in generated images, describe several implementation details for balancing training and suggest a new metric for evaluating GAN results.

### Progressive Growing of GANs

![](img/pg_gan_growing.png)

The incremental nature allows the training to first discover large-scale structure of the image distribution and then shift attention to increasingly finer scale detail, instead of having to learn all scales simultaneously. 

![](img/pg_gan_fakeIn.png)

When new layers are added to the networks, we fade them in smoothly, to avoid sudden shocks to the already well-trained, smaller-resolution layers.

the progressive training has several benefits:
1) the generation of smaller images is substantially more stable because there is less class information and fewer modes
2) With progressively growing GANs most of the iterations are done at lower resolutions, and comparable result quality is often obtained up to 2–6 times faster

### Increasing Variation
GANs have a tendency to capture only a subset of the variation found in training data. The authors propose a simplified solution:
1) first compute the standard deviation for each feature in each spatial location over the minibatch
2) average these estimates over all features and spatial locations to arrive at a single value
3) replicate the value and concatenate it to all spatial locations and over the minibatch, yielding one additional(constant) feature map. 

![](img/pg_gan_ministd.png)

### Normalization in Generator and Discriminator
GANs are prone to the escalation of signal magnitudes as a result of unhealthy competition between the two networks. We use a different approach instead of  batch normalization to constrain signal magnitudes and competition, because the normalization methods were originally introduced to eliminate covariate shift.

Equalized Learning Rate
To be precise, we set w'(i) = w(i)/c, where w(i) are the weights and c is the per-layer normalization constant from He’s initializer. 

![](img/pg_gan_equalized.png)

These methods normalize a gradient update by its estimated standard deviation, thus making the update independent of the scale of the parameter.

Pixelwise Feature Vector Normalization in Generator

To disallow the scenario where the magnitudes in the generator and discriminator spiral out of control as a result of competition, we normalize the feature vector in each pixel to unit length in the generator after each convolutional layer.

![](img/pg_gan_pixelwise_norm.png)

### Multi-Scale Statistical Similarity for Assessing GAN Results
The authors argue that existing methods such as MS-SSIM find large-scale mode collapses reliably but fail to react to smaller effects such as loss of variation in colors or textures, and they also do not directly assess image quality in terms of similarity to the training set.

The authors propose to study this by considering the multiscale statistical similarity between distributions of local image patches drawn from Laplacian pyramid representations of generated and target images, starting at a low-pass resolution of 16 × 16 pixels.
1) randomly sample 16384 images and extract 128 descriptors from each level in the Laplacian pyramid, giving us 221 (2.1M) descriptors per level. We denote the patches from level l of the training set and generated set as {x} and {y} respectively. 
2) first normalize {x} and {y} w.r.t. the mean and standard deviation of each color channel
3) and then estimate the statistical similarity by computing their sliced Wasserstein distance SWD, an efficiently computable randomized approximation to earthmovers distance, using 512 projections. 

### Experiments
importance of individual contributions in terms of statistical similarity

![](img/pg_gan_exp.png)

convergence and training speed

![](img/pg_gan_exp_speed.png)
