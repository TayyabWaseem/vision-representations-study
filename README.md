# Visual Representation Learning with Deep Neural Networks (CS 6304)

A series of experiments studying how different deep learning architectures learn and represent visual information.

## Experiments

### ResNet-152
Experiments on CIFAR-10 using an ImageNet-pretrained ResNet-152:
- Baseline transfer learning with a frozen backbone
- Ablation of residual skip connections
- Feature visualization across early, middle, and late layers using t-SNE
- Comparison of pretrained and randomly initialized models

### Vision Transformer (ViT)
Experiments investigating:
- ImageNet classification predictions
- Attention heatmaps
- Random and center patch masking
- Comparison of `[CLS]` and mean-pooled patch representations

### CLIP
Experiments using OpenAI's pretrained CLIP model to study:
- Zero-shot image classification
- Image-text modality representations
- The image-text modality gap
- Alignment between image and text embeddings

### Variational Autoencoder (VAE)
A VAE trained on MNIST with a 20-dimensional latent space. The analysis includes:
- Latent-space visualization using t-SNE
- Image reconstruction
- Generation from the standard normal prior
- Latent interpolation
- KL-divergence analysis

## Repository Structure

```text
vision-representations-study/
├── notebooks/
│   ├── resnet-1.ipynb
│   ├── vit-2.ipynb
│   ├── clip-3.ipynb
│   └── vae-4.ipynb
├── outputs/
│   ├── resnet/
│   ├── vit/
│   ├── clip/
│   └── vae/
├── utils/
│   ├── file_utils.py
│   └── model_utils.py
├── requirements.txt
└── Visual_Representation_Learning_with_Deep_Neural_Networks.pdf