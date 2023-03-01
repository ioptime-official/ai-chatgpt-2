# ai-chatgpt-2
### Introduction

- In this project I have selected medical dataset and GPT2 model for the text genration
- We provide some text the model on the base of text generate results
### Project Workflow

- Firstly Start by collecting the data related to the domain. In this project, healthcare-related data and reports [Gather](https://github.com/LasseRegin/medical-question-answer-data.git) to train a GPT2 generative model. 
- Then acquired data is converted to .csv format and removed any unwanted characters or tags from it.Apply some preprocessing techniques such as removing duplicates and outliers, handling missing data, and adding limits to the maximum number of words in a sentence.
- Use the Tokenizer class provided by the Hugging Face Transformers library to tokenize the text into subwords. We also apply padding and add special start and end tokens.
- Load the preprocessed data into PyTorch DataLoader to split it into batches. Use the pre-trained GPT2 model provided by the Transformers library to train on our preprocessed data.
-dAfter the model training is completed, evaluate the performance of the model by using metrics like loss function and perplexity.Calculate the performance metrics to determine the quality of the model.
- Finally, use the trained model to generate new text sequences related to the healthcare domain. Download the generated model for future use.


![](https://github.com/ioptime-official/ai-chatgpt-2/blob/main/workflow.jpg)


###  Dependencies:
```
transformers==4.11.3
torch==1.9.0
numpy==1.19.5
pandas==1.2.4
matplotlib==3.3.4
```
these dependecies can be download by runing given command
`pip install -r requirements.txt`



### Generate sentences!
After training GPT-2, we can generate sentences with our trained model using.
```
model.eval()

prompt = "i am feeling pain in shoulder muscle "

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
# generated = generated.to(device)

print(generated)

sample_outputs = model.generate(
                                generated, 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=50, 
                                max_length = 100,
                                top_p=0.95, 
                                num_return_sequences=1,
                                # temperature=1,
                                )

for i, sample_output in enumerate(sample_outputs):
  print(sample_output)
  print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
  result = "{} : {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True))

```
After building the model the folder hierarchy look like.
```
├── src
│   ├── dataset
│   │   ├── data.csv
|   ├── model
│   |   ├── pytorch_model.bin
│   |   ├── tokenizer_config.json
│   |   ├── merge.txt
│   |   ├── added_tokens.json
│   |   ├── config.json
│   |   ├── vocab.json
```
### Results:
I have following results after passing keyword to model
- Keywords: "symptoms of calcium deficiency"
> symptoms of calcium deficiency are numerous they include fatigue dizziness irritability weakness fatigue and weakness in addition symptoms appear like flu like symptoms in which someone has a low level of circulating acid and nutrients including vitamin d calcium calcium magnesium and sodium it is important to note calcium is essential for brain growth and joint movement if a person has not exercised regularly enough he or she may suffer from mild to severe weakness
- Keywords: "causes of lung cancer"
>  causes of lung cancer vary on a person is known risk factors to their immediate surroundings such as family history of cancer or certain health conditions a person makes an important lifestyle change most changes will be absorbed from their lungs in gradual process and are in good to moderate risk this process helps minimize the possibility for contagious diseases through a smoke or chemical exposure when smoking or environmental exposures like smoke gums or particulates become widespread an immediate medical evaluation such as h pylori positive a team of disease experts has observed that



