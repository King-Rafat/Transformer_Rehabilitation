# Transformer Network for the Assessment of Physical Rehabilitation Exercise

* Accepted at **IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE 2025)** | [Paper](https://aclanthology.org/2023.emnlp-main.618/) | [Model]() | [Dataset](https://huggingface.co/datasets/PeacefulData/HyPoradise-v1-GigaSpeech)
* We use a Curve-based Data Aggregation for feature augmentation then fuse it with a Transformer architecture to assess quality of rehabilitation rehabilitation exercise.


![Screenshot](https://github.com/King-Rafat/Transformer_Rehabilitation/blob/main/Images/First_Diagram.pdf)


## Results
### 

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://github.com/King-Rafat/Transformer_Rehabilitation/blob/main/Images/First_Diagram.pdf)

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

