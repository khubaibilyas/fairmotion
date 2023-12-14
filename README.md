## CS 7643 Deep Learning

**Team**: Motion Prediction

**Members**: Ibrahim Abdulhafiz, Khubaib Ilyas, Karla Saillant, and Matthias Stefan Schnabl

*This is a fork of the Facebook Fairmotion library which we used as the benchmark for our project.
Reference [original README](README-facebook.md) for licensing information and additional details.*

## Cloning

```shell script
git clone https://github.com/iabdulhafiz/fairmotion.git
```

## Dataset Download

We downloaded the Synthetic 60FPS zip from the [AMASS DIP site](https://dip.is.tuebingen.mpg.de/).

## Preprocessing, Training, and Validation

We mainly referenced the provided [readme under the motion prediction task section](https://github.com/iabdulhafiz/fairmotion/blob/main/fairmotion/tasks/motion_prediction/README.md).
This README contains steps on preprocessing, training, and validation.

Alternatively, to setup the environment, Google Colab Pro can be used with an increased System RAM and GPU.
Members of the team that did not have the necessary GPU to run locally used a Google Colab Jupyter notebook similar to [this one](fairmotion-colab.ipynb)

## Custom Architectures

### Transformer Decoder Architecture

See Git history for the difference in file changes made to create this architecture.

### Split Joint Architecture

See the `rnn_split_joint_architecture` [branch](https://github.com/iabdulhafiz/fairmotion/tree/rnn_split_joint_architecture) to view the changes made to create this architecture.

The following command was used to train this custom model on the above branch.

```shell script
python /content/fairmotion/fairmotion/tasks/motion_prediction/training.py \
    --save-model-path '/content/drive/MyDrive/7643/runs-rnn/' \
    --preprocessed-path '/content/drive/MyDrive/7643/aa' \
    --epochs 100 --architecture rnn \
    --batch-size 32 \
    --save-model-frequency 2 \
    --hidden-dim 240 \
    --split_architecture True
```
