# CapsComplete

## Installation

```bash
pip install .
```

## Data


## Creating a Model
Any occlusion model must contain the following:  

**Inputs/Placeholders, Local Variables**
  * Constructor Input: `(height, width, channels, num_label, is_training, **kwargs)`
  * `.inputs` Input placeholder of size `[-1, height,width,channels]`
  * `.labels` Label input placeholder of size `[-1,]`
  * `.graph` Graph to call session with

**Operations**
  * `.train_op`: Training step
  * `.total_loss`: Some variant typically involving classification and reconstruction loss
  * `.classification_loss`: Classification Loss typically softmax_cross_entropy
  * `.accuracy`: Accuracy Operation
  * `.train_summary`: Summary operation
  * `.probs`: Probability layer for each class
  * `.predictions`: Predictions
  * `.recons`: Reconstructed images in original size padded with first dimension

Current models are:
  * `AutoEncoder`
  * `CapsNet`
  * `AutoEncoder Gradient`
  * `CapsNet Gradient`

## Evaluation
