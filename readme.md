# Attention Is All You Need

It is a pytorch implementation of self-attention or transformer from scratch. In this implementation, each word is predicted rather than each character.

![alt text](images/transformer_architecture.png)

The image above shows the architecture of the transformer. It esssentailly consists of two blocks:
1. Encoder
2. Decoder

## Encoder Blocks:
It is used to encode all the time information into a single vector. If we look closely at the block, the encoder consists of these sections:
1. Input Encoding: The entire training and validation dataset is used to create a dictionary of words and a unique ID is assigned to each word. This was achieved by using tokenizer package from hugging face. This unique ID is then converted to to feature space of size 'd_model'. The model learns this conversion during trianing.

2. Positional Encoding:  'sin' and 'cos' positional encoding are added to the input encoding before starting the inference/training. This encoding is constant and not learnt and is used to make the model understand the position of each word in a sentence.

![alt text](images/encodings.png)

3. Self-attention: One way to understand iis that, correlation scores are calculated between the input words, i.e. how much is a word related to another word in the input. To calculate the score, following formula is used:

attention(K,Q,V) = softmax $(\dfrac{K @ Q^{T}}{\sqrt{d_{model}}}).V$

where K, Q, V represents Key, Query and Value