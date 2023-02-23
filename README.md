# ai-chatgpt-2
### Introduction

- In this project I have chose medical dataset and GPT2 model for the text genration
- We provide some text the model on the base of text generate results

###  Dependencies:
- Transformer
- torch
- numpy 
- pandas
- matplotlib
these dependecies can be download by runing given command
`$ pip install -r requirements.txt`



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
