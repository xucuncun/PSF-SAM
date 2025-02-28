# PSF-SAM
PSF-SAM is a model designed for colon polyp segmentation, aiming to enhance performance through adaptive feature learning and sample selection mechanisms. This model combines deep learning with task-specific optimization strategies to significantly improve segmentation accuracy.

## Training and Freezing Specific Layers

During training, the parameters of the original model can be frozen by calling the freeze_except function from the utility:

`net = freeze_except(net, ['polyp_feature', 'Domain_generalization', 'Adapter_mlp', 'linear_classic'])`  

This function freezes all layers of the model except for the specified ones, making fine-tuning more efficient by preventing unnecessary updates to pre-trained weights.
