# Monet CycleGAN: Style Transfer from Photos to Monet's Impressionist Paintings

## Problem Description and Dataset

This project addresses the challenge of transforming ordinary photographs into images that mimic Claude Monet's distinctive impressionist painting style using a CycleGAN model. The dataset consists of two distinct collections: approximately 300 Monet paintings and over 7,000 photographs with similar landscape themes. This significant class imbalance (1:23 ratio) presents an interesting challenge for the model training process.

Both image sets have been standardized to 256x256 pixel resolution with RGB color channels. File size distributions are similar between datasets, with Monet paintings averaging 12.5-17.5 KB and photos averaging 15-17.5 KB. The Monet paintings showcase the artist's characteristic impressionist style with visible brushstrokes, emphasis on light effects, and subtle color variations, while the photographs represent realistic depictions of similar subjects.

## Exploratory Data Analysis

Initial exploration revealed several key insights about the dataset:

- **Class Imbalance**: The significant disparity between Monet paintings (300) and photos (7,000+) necessitated data augmentation strategies specifically for the Monet class to prevent bias.

- **Image Properties**: Analysis of image dimensions, color channels, and file formats confirmed consistent preprocessing, with all images standardized to 256x256 RGB format in JPEG compression.

- **Color Distribution**: Comparative color channel analysis showed distinctive differences in RGB distributions between Monet paintings and photographs, with Monet works having more pronounced blues and warmer tones characteristic of impressionist style.

- **Visual Inspection**: Random sampling of images from both classes highlighted the transformation challenge - converting sharply detailed, diverse photographs into images with Monet's distinctive brushwork, atmospheric effects, and impressionist lighting.

Based on this analysis, we implemented data augmentation for the Monet class (including horizontal flips, brightness and contrast adjustments) and developed a train-validation split strategy to properly evaluate model performance.

## Model Architecture

The project implements a CycleGAN architecture with the following components:

### Generators
- Two ResNet-based generators: one for Photo→Monet transformation and one for Monet→Photo transformation
- Each generator follows identical architecture with shared weights

#### Generator Architecture Table

| Layer (type)                 | Output Shape          | Param #    | Connected to        |
|------------------------------|------------------------|------------|---------------------|
| input_layer (InputLayer)     | (None, 256, 256, 3)   | 0          | -                   |
| conv2d (Conv2D)              | (None, 256, 256, 64)  | 9,408      | input_layer[0][0]   |
| instance_normalization       | (None, 256, 256, 64)  | 128        | conv2d[0][0]        |
| leaky_re_lu (LeakyReLU)      | (None, 256, 256, 64)  | 0          | instance_norm[0][0] |
| conv2d_1 (Conv2D)            | (None, 128, 128, 128) | 73,728     | leaky_re_lu[0][0]   |
| instance_normalization_1     | (None, 128, 128, 128) | 256        | conv2d_1[0][0]      |
| leaky_re_lu_1 (LeakyReLU)    | (None, 128, 128, 128) | 0          | instance_norm_1[0][0] |
| conv2d_2 (Conv2D)            | (None, 64, 64, 256)   | 294,912    | leaky_re_lu_1[0][0] |
| instance_normalization_2     | (None, 64, 64, 256)   | 512        | conv2d_2[0][0]      |
| leaky_re_lu_2 (LeakyReLU)    | (None, 64, 64, 256)   | 0          | instance_norm_2[0][0] |
| ...                          | ...                   | ...        | ...                 |
| [9 residual blocks with 2 convolutions each and skip connections] | | | |
| ...                          | ...                   | ...        | ...                 |
| conv2d_transpose (ConvTranspose2D) | (None, 128, 128, 128) | 294,912  | add_8[0][0]        |
| instance_normalization_19    | (None, 128, 128, 128) | 256        | conv2d_transpose[0][0] |
| re_lu (ReLU)                 | (None, 128, 128, 128) | 0          | instance_norm_19[0][0] |
| conv2d_transpose_1 (ConvTranspose2D) | (None, 256, 256, 64) | 73,728 | re_lu[0][0]        |
| instance_normalization_20    | (None, 256, 256, 64)  | 128        | conv2d_transpose_1[0][0] |
| re_lu_1 (ReLU)               | (None, 256, 256, 64)  | 0          | instance_norm_20[0][0] |
| conv2d_21 (Conv2D)           | (None, 256, 256, 3)   | 9,411      | re_lu_1[0][0]       |
| activation (Activation)      | (None, 256, 256, 3)   | 0          | conv2d_21[0][0]     |

**Total parameters**: 8,406,147  
**Trainable parameters**: 8,406,147  
**Non-trainable parameters**: 0

The architecture follows a U-shaped design with:
1. **Initial processing**: 7×7 convolutional layer with 64 filters
2. **Downsampling path**: Two blocks that reduce spatial dimensions while increasing feature channels (256×256×64 → 128×128×128 → 64×64×256)
3. **Residual blocks**: Nine identical blocks at the bottleneck, each containing two convolutions and a skip connection
4. **Upsampling path**: Two transposed convolutions to restore original dimensions (64×64×256 → 128×128×128 → 256×256×64)
5. **Output layer**: 7×7 convolution with 3 filters and tanh activation

The generator contains approximately 8.4 million parameters, with most concentrated in the residual blocks. The skip connections in these blocks are crucial for maintaining gradient flow through this deep architecture and preserving spatial information throughout the network.

### Discriminators
- Two PatchGAN discriminators for adversarial training
- Each discriminator uses:
  - Four convolutional layers with increasing filter counts
  - LeakyReLU activations
  - Instance normalization (custom implementation for TF 2.19 compatibility)

### Loss Functions
- Adversarial Loss: Least Squares GAN (LSGAN) loss for stable training
- Cycle Consistency Loss: Enforces x → G(x) → F(G(x)) ≈ x, preserving content
- Identity Loss: Encourages generators to preserve input colors when appropriate

This architecture was selected for several reasons:
1. **CycleGAN's unpaired training capability** matches our scenario where direct pairs of Monet-photo images don't exist
2. **ResNet blocks** enable deeper networks while maintaining gradient flow
3. **Instance normalization** performs better than batch normalization for style transfer
4. **PatchGAN discriminators** focus on local image structure rather than global features
5. **Multiple loss functions** balance adversarial training with content preservation

The implementation includes a custom InstanceNormalization layer for TensorFlow 2.19 compatibility, replacing the TensorFlow Addons dependency in the original CycleGAN paper.

## Results and Analysis

The model was trained for multiple epochs with performance evaluated using both standard metrics and the specialized MiFID (Memorization-informed Fréchet Inception Distance) metric required by the Kaggle competition.

### Training Performance
- **Generator Losses**: Both Monet and Photo generators showed steady convergence, starting at ~7.8 and decreasing to ~6.0 over 9 epochs
- **Discriminator Losses**: Similarly decreased from 0.40-0.45 to approximately 0.20, indicating balanced GAN training
- **Training Stability**: The model maintained balance between generators and discriminators, avoiding common GAN issues like mode collapse

### Evaluation Metrics
- **FID Score**: Remained relatively stable around 300, indicating consistent visual quality
- **MiFID Score**: Showed an increasing trend from ~800 to ~850, suggesting potential memorization issues
- **Visual Assessment**: Generated images demonstrated recognizable Monet stylistic elements, though with varying quality

### Hyperparameter Tuning
Several parameters were experimented with:
- **Cycle Consistency Weight (λ)**: Values between 5-10 balanced content preservation with style transfer
- **Learning Rate**: 2e-4 with Adam optimizer (β1=0.5) provided stable training
- **Batch Size**: Small batch size (1) proved most effective for this task
- **Residual Blocks**: 9 blocks provided sufficient depth without excessive parameters

### Challenges and Solutions
- **GPU Memory Constraints**: Could not use. Could change to PyTorch
- **TensorFlow 2.10 Compatibility**: Resolved by implementing custom InstanceNormalization layer
- **Optimizer Variable Tracking**: Fixed by recreating optimizers after initial variable creation
- **Class Imbalance**: Mitigated through data augmentation and balanced dataset creation

## Conclusion

The implemented CycleGAN model successfully demonstrates the capability to transform photographs into Monet-style paintings, capturing many of the impressionist characteristics found in Monet's work. The stable training progression and consistent FID scores indicate that the model effectively learned aspects of the style transfer task.

However, the increasing MiFID score suggests room for improvement in preventing the model from simply memorizing training examples rather than generalizing Monet's style. This highlights the importance of using specialized metrics like MiFID that account for memorization when evaluating generative models.

Several potential improvements for future work include:
1. More extensive regularization techniques to prevent memorization
2. Exploration of alternative loss formulations specifically targeting MiFID optimization
3. Curriculum learning approaches that gradually introduce more complex style elements
4. More sophisticated data augmentation for the Monet class to further address class imbalance
5. Extended training with learning rate scheduling to fine-tune the models

Overall, this project demonstrates both the capabilities and limitations of using CycleGAN for artistic style transfer, providing valuable insights into GAN training dynamics and evaluation metrics for generative models.
