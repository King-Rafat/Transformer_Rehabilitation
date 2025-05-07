# Transformer Network for the Assessment of Physical Rehabilitation Exercise

* Accepted at **IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE 2025)** | [Paper](https://aclanthology.org/2023.emnlp-main.618/) | [Model]() | [Dataset](https://huggingface.co/datasets/PeacefulData/HyPoradise-v1-GigaSpeech)
* We use a Curve-based Data Aggregation for feature augmentation then fuse it with a Transformer architecture to assess quality of rehabilitation rehabilitation exercise.


![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.]()
Figure 1: Illustration of carbon footprints used by different deep models while (a) training on CIFAR 100 (in log scale) and (b-c) inferring on evaluation set. ResNet18 is a deeper model with 11.2M parameters, resulting in higher inference time (4.7 sec.) and CO2 emission (0.087 g). To minimize this, using ResNet18 as a teacher, we train two student models, MobileNetV2 (student 1) and ShuffleNetV2 (student 2), following the traditional KD process. This training costs significant carbon footprints (red and green dashed curves in (a)) with an accuracy increment from learning the teacher model (black dotted curve in (a)). However, as expected, both students consume less time and CO2 during inference (red and green shaded bars in (b) and (c)). We aim to reduce the training cost and CO2 production of the KD process while using the same students (red and green solid curves in (a)) and maintain similar accuracy and inference costs (solid red and green bars in (b) and (c)) in comparison with the costly KD training.

## Results
### 

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.]()

### Dependencies
- python>=3.7
- pytorch, cudatoolkit>11.2 (Might work for lower)

### Datasets

* UI-PRMD: 

Paper link: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

Website: https://www.cs.toronto.edu/~kriz/cifar.html

File: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

* KIMORE: 
Paper link: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

Website: https://www.cs.toronto.edu/~kriz/cifar.html

File: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

* IRDS: 

Paper link: http://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf

File: http://cs231n.stanford.edu/tiny-imagenet-200.zip

### Dataset Split



### Files

* `Rehabilitation.ipynb` : contains code to perform training, inference, validation, generate visualization, and contains model architecture
* `Data_Processing` : consists of code to load the KIMORE dataset
* `core` : consists of code for Curve-Net (architecture) for curve based point aggregation

### How to run
- 

### Used Repositories

* Loading KIMORE dataset: [KiMoRe_wrapper](https://github.com/petteriTeikari/KiMoRe_wrapper)
* Processing KIMORE dataset: [STGCN](https://github.com/fokhruli/STGCN-rehab)
* Loading and Processing UI-PRMD dataset: [UI-PRMD](https://github.com/avakanski/A-Deep-Learning-Framework-for-Assessing-Physical-Rehabilitation-Exercises)

## Cite This Paper
If you use this code and model and dataset splits for your research, please consider citing:

