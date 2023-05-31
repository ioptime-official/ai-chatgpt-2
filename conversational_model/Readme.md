
###  Conversational Healthcare Dataset
This repository contains a conversational healthcare dataset used for training and testing purposes. The dataset consists of an English dataset comprising 0.26 million conversations, 0.51 million utterances, and 44.53 million tokens. It covers 96 specialties of diseases, providing a comprehensive collection of healthcare-related conversations.

### Dataset Details
- Conversations: 0.26 million
- Utterances: 0.51 million
- Tokens: 44.53 million
- Specialties: 96

Data can be downlaoded from this [link](https://drive.google.com/drive/folders/17QXf5kyMVdzZSoG6gX9hqSk8IETbbrwe?usp=sharing).

### Distributed Training
The model was trained using a distributed training approach, utilizing 2 devices. The dataset was split and distributed across these devices, enabling parallel training and improving efficiency. Kaggle, a popular platform for data science and machine learning, was utilized for the distribution training process.


###  Model Files:
After the model trainig model files saved ,these files can be access from [here](https://drive.google.com/drive/folders/123ndbHDXSixzwvsRyjROwl7AI_Eboapu?usp=sharing)

###  Dependencies:

before runing the code follwing required libraries must be installed 
```
transformers
torch
numpy
pandas

```
These dependecies can be download by runing given command
`pip install -r requirements.txt`

