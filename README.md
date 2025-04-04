# PCHGCN-Master

The code of PCHGCN: Physically Constrained Higher-order Graph Convolutional Network for Pedestrian Trajectory Prediction

The Paper (Early Access): [PCHGCN: Physically Constrained Higher-order Graph Convolutional Network for Pedestrian Trajectory Prediction](https://ieeexplore.ieee.org/document/10948459)

The code and weights have been released, enjoy it！ You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the models performance run:  test.py

The specific code of the training part will be released after the paper is officially published!

# PCHGCN-A brief introduction to the overall architecture
Higher-order graphs can effectively model indirect higher-order social relations between pedestrians, but it is inevitable to bring excessive redundant interactions into the modeling of higher-order graphs. To reasonably describe social relationships of different orders, we propose a physically constrained higher-order graph convolutional network. Specifically, we first construct a spatial graph and utilize the attention mechanism to obtain a spatial attention score matrix for the preliminary representation of pedestrian social interactions. At the same time, we input the history trajectories into the physical constraint module, which determines whether there is an interaction based on physical characteristics such as field of view, distance, and distance transformation rate, thereby generating a physical mask matrix. The spatial graph, spatial attention score matrix, and physical mask matrix are then passed into the higher-order graph module to accurately capture the social interaction features of various orders. Subsequently, the weighted fusion of each-order social interaction features is realized through the gated fusion module. Finally, the temporal convolutional networks (TCNs) predict the offset between the CVM-predicted trajectory and the ground truth to achieve multimodal future trajectory prediction.

<img width="725" alt="Figure 3 - 修改3" src="https://github.com/user-attachments/assets/01e2a67f-34c4-4f90-a107-12d8b758ba0b" />


## Code Structure
checkpoint folder: contains the trained models

dataset folder: contains ETH, UCY and SDD datasets

model.py: the code of PCHGCN

test.py: for testing the code

utils.py: general utils used by the code

metrics.py: Measuring tools used by the code

## Model Evaluation
You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the models performance run:  test.py

## Acknowledgement
Some codes are borrowed from Social-STGCNN, SGCN, IMGCN and DSTIGCN. We gratefully acknowledge the authors for posting their code.
