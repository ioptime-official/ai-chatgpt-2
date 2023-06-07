
###  Conversational and Descriptive  Healthcare Dataset
This repository contains a conversational and descriptive healthcare dataset used for training and testing purposes. The dataset consists of an English dataset comprising 0.28 million conversations, 0.51 million utterances, and 44.53 million tokens. It covers 96 specialties of diseases, providing a comprehensive collection of healthcare-related conversations.

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


### Inference:
Model can be tested after downloading the files from the model files and run following code
```
DEBUG           = False


USE_APEX        = True
APEX_OPT_LEVEL  = 'O1'

MODEL           = 'gpt2' #{gpt2, gpt2-medium, gpt2-large, gpt2-xl}

UNFREEZE_LAST_N = 6 #The last N layers to unfreeze for training

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
                    
MAXLEN          = 256  #{768, 1024, 1280, 1600}

TRAIN_SIZE      = 0.8

if USE_APEX:
    TRAIN_BATCHSIZE = 16
    BATCH_UPDATE    = 5
else:
    TRAIN_BATCHSIZE = 8
    BATCH_UPDATE    = 8

EPOCHS          = 5
LR              = 5e-4
EPS             = 1e-8
WARMUP_STEPS    = 1e2

SEED            = 2020
os.environ['WANDB_DISABLED'] = 'true'
# Tokenizer and model function
def get_tokenier(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer

def get_model(tokenizer, special_tokens=None, load_model_path=None):

    #GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    #----------------------------------------------------------------#
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    model.cuda()
    return model
    
    
tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer, 
                  special_tokens=SPECIAL_TOKENS,
                  load_model_path='place the pytorch.bin file address')
                  
                  
title = "Place your Query here"
prompt = SPECIAL_TOKENS['bos_token'] + title + SPECIAL_TOKENS['sep_token'] 
         
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
device = torch.device("cuda")
generated = generated.to(device)

model.eval();

sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                min_length=50, 
                                max_length=MAXLEN,
                                top_k=30,                                 
                                top_p=0.7,        
                                temperature=0.9,
                                repetition_penalty=2.0,
                                num_return_sequences=10
                                )

for i, sample_output in enumerate(sample_outputs):
    text = tokenizer.decode(sample_output, skip_special_tokens=True)
    a = len(title)  
    print("{}: {}\n\n".format(i+1,  text[a:]))


```



