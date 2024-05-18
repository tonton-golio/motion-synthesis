import streamlit as st


# Title and intro
"""# Transformer from Scratch
*In this page, we will implement a transformer from scratch. And use it to generate Shakespearean text.*
"""
with st.expander("Transformer from Scratch"):
    st.write("""


Based on the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al., the Transformer model is a novel neural network architecture that has been proven to be more efficient than the traditional RNNs and LSTMs in sequence-to-sequence tasks. The Transformer model is based on the self-attention mechanism, which allows the model to focus on different parts of the input sequence when predicting each part of the output sequence.

The model was initially designed for machine translation tasks, but it has been successfully applied to various other tasks, such as text generation, summarization, and question answering.

I will be following alonmg with Andrej Karpathy's video: https://www.youtube.com/watch?v=kCc8FmEb1nY

Attention Is simply a communication mechanism which allows? features corresponding to different time steps in our sequence to communicate with each other Typically, we don't let Time steps in the future communicate with time steps in the past as we are trying to predict the future.

The central formula in the transformer is the scaled dot product attention:
$$
    Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
scaled refers to the division by squareroot head-size or of C???. This controls the variance. Softmax converges towards one-hot encodings, if applied on vectors containing large absolute values. This is why we divide by the square root of the dimension of the key vectors. This ensures that the softmax is not too extreme.

We implement the self attention mechanism. Self attention means that. the keys queries and values are all generated from the same data source (the input sequence). The self attention mechanism allows the model to focus on different parts of the input sequence when predicting each part of the output sequence. Cross attention on the other hand. Lets keys queries and values come from arbitrary sources.

We can instead do multi-head attention; which is just the concatenation of the output of multiple attention heads, followed by a linear layer. This allows the model to focus on different parts of the input sequence in parallel. The output of the multi-head attention is then passed through a feedforward neural network.

We also need skipped connections (aka residual connections) as per (https://arxiv.org/abs/1512.03385). They introduced the concept of skipped conenctions with addition. Addition distributes gradients equall to both of its branches.

Once we get to a couple blocks of multihead attention, it becomes a big network. Large networks are hard to train, so we need to add layer norm. Its similar to batchnorm, in batch norm we make sure that across the batch dimension, we have zero mean and unit variance. Layernorm normalizes across rows instead of columns.

To improve the model further, and prevent overfitting, we can add dropout. Dropout is a regularization technique that randomly sets a fraction of the input units to zero during training. This helps prevent overfitting and improves the model's generalization capabilities. By using drop out, we force the network to learn redundant representations of the data, which helps prevent overfitting. (source: https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

What we have built here; is a decoder-only transformer. This architecture differs from that proposed in the original paper. In Attention is all you need, they use a encoder decoder architecture. This is because they wanted to do machine translation, for which the output sequence is dependent on some conditioning input. In our case, we are doing language modeling, so we don't need an encoder. We can just use the decoder part of the transformer.

Looking forward to our diffusion model; if we choose to go with a transformer, it will make sense to expand to a a full transformer model, such that we can supply conditioning (i.e., time-step and text-description).
""")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def load(verbose=False):
    # load data from file
    with open('assets/tinyShakespear.txt') as f: text = f.read()
    if verbose:
        print("length of text:", len(text))
        print('First 100 characters:', text[:100]+'...')
    return text

def lang_setup(text, verbose=False):
    # set up vocabulary and mapping
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    if verbose:
        print('Vocabulary size:', vocab_size)
        print(chars)

    # we need mapping from char to index and index to char
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda x: [stoi[ch] for ch in x]
    decode = lambda x: ''.join([itos[i] for i in x])
    if verbose:
        encoded_text = encode('This is a test')
        print('encoded_text:', encoded_text)
        print('decoded_text:', decode(encoded_text))
    return vocab_size, encode, decode

def test_val_split(text, encode, split=0.9):
    # make data tensor
    X = torch.tensor(encode(text))
    # train test split
    n = int(split * len(X))
    X_train, X_val = X[:n], X[n:]

    return X_train, X_val

def batchify(X, block_size, batch_size, device):
    # get random starting points for each batch
    ix = torch.randint(X.size(0) - block_size, (batch_size,)).to(device)  # random starting indices
    # construct the batch
    # print(ix)
    x = torch.stack([X[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([X[i+1:i+block_size+1] for i in ix]).to(device) # y is the same as x, but shifted one position to the right
    return x, y

class FeedForward(nn.Module):
    def __init__(self, n_embed=32, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed*4),
            nn.ReLU(),
            nn.Linear(n_embed*4, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    """Self attention mechanism for the transformer model."""
    def __init__(self, head_size=16, n_embed=32,  dropout=0.1,):
        super(SelfAttention, self).__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # lower triangular matrix, so we only see the past

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C) -> (B, T, head_size)
        q = self.query(x)  # (B, T, C) -> (B, T, head_size)

        # query, key, value
        
        wei = q@k.transpose(-2, -1)  # initialize weights, affinity, or attention
                                    # instead of zeros, wei should be the dot product of query and key
                                    # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # set the future to -inf, so we dont attend to it
        wei /= C**0.5  # scale by sqrt of C
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        xbow = wei @ v  # (T, T) @ (B, T, C) -> (B, T, C), apply matrix multiplication to each batch in parallel
        # we do a simple average of the past tokens and current token
        return xbow

class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads=8, head_size=16, dropout=0.1,):
        super(MultiHeadAttention, self).__init__()
        n_embed = n_heads * head_size
        self.heads = nn.ModuleList([SelfAttention(head_size, n_embed, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)  # projection back into the residual pathway
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        print('step_b1')
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        print('step_b2')
        out = self.proj(out)
        print('step_b3')
        out = self.dropout(out)
        print('step_b4')
        return out

class AttentionBlock(nn.Module):
    """
    Interspersed feedforward layer with multi-head self attention mechanism
    """
    def __init__(self, n_embed=32, n_head=8, dropout=0.1,):
        super(AttentionBlock, self).__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)


    def forward(self, x):
        print('step_a1')
        x = self.ln1(x)
        print('step_a2')
        x = x + self.sa(x)
        print('step_a3')
        x = self.ln2(x)
        print('step_a4')
        x = x + self.ffwd(x)  # the plus here is our residual connection
        print('step_a5')
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed=32, n_head=16, n_layer=3,):
        super(LanguageModel, self).__init__()

        assert n_embed % n_head == 0, 'Embedding dimension must be divisible by number of heads'

        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[AttentionBlock(n_embed, n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.criteria = nn.CrossEntropyLoss()
    
    def forward(self, x, y=None):
        B, T = x.shape
        print('step1')
        tok_emb = self.token_embedding(x)  # shape [batch_size, block_size, n_embed]
        pos_emb = self.positional_embedding(torch.arange(T, device=x.device))
        print('step2')
        x = pos_emb + tok_emb
        print('step3')
        x = self.blocks(x)
        print('step4')

        logits = self.lm_head(x)  # shape [batch_size, block_size, vocab_size]
        # logits are scores for next token at each position
        print('step5')
        if y is None: 
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            print('step6')
            print(logits.device, y.device)
            print(logits.shape, y.shape)
            print(logits.dtype, y.dtype)
            # loss = F.cross_entropy(logits, y.view(B*T))  # negative log likelihood loss
            loss = self.criteria(logits, y)
            print('step7')
        return logits, loss
    
    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            x_ = x[:, -block_size:]
            logits, _ = self.forward(x_)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
        return x

def estimate_loss(model, X, block_size, batch_size, device, eval_iter=100):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iter):
            x, y = batchify(X, block_size, batch_size, device)
            _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
    model.train()
    return torch.tensor(losses).mean().item()

def train():
    text = load()
    vocab_size, encode, decode = lang_setup(text)
    X_train, X_val = test_val_split(text, encode)


    # now for the transformer model
    

    # create model    
    m = LanguageModel(vocab_size, n_embed=n_embed, n_head=n_head, n_layer=n_layer).to(device)
    print('Number of parameters:', sum(p.numel() for p in m.parameters()))

    # training
    optimizer = optim.AdamW(m.parameters(), lr=0.001)
    n_iter = 5000
    print('training...')
    for i in range(n_iter):
        m.train()
        x, y = batchify(X_train, block_size, batch_size, device)
        # print(x.device, y.device)
        _, loss = m(x, y)
        # print(x.device, y.device, loss.device)
        # assert False
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'iteration {str(i).rjust(5)}, loss: {loss.item():.2f}, val loss: {estimate_loss(m, X_val, block_size, batch_size, device):.2f}')

    # save model
    torch.save(m.state_dict(), 'assets/shakespear_model.pth')
    return m, decode

def load_model():
    text = load()
    vocab_size, encode, decode = lang_setup(text)
    # load model
    m = LanguageModel(vocab_size, n_embed=n_embed, n_head=n_head, n_layer=n_layer).to(device)
    m.load_state_dict(torch.load('assets/shakespear_model.pth'))
    return m,  decode, encode

torch.manual_seed(1337)
device = torch.device('cpu')
block_size = 16  # even though we have a block size of 8, the model will take context of 1-8 tokens when predicting the next token
batch_size = 64
n_embed = 64
n_head = 4
n_layer = 4

train()
m, decode, encode = load_model()

out = m.generate(torch.tensor(encode('To be or not to be, that is the question:')).unsqueeze(0), 100)

out

# inference
context = torch.zeros((1,1), dtype=torch.long)
st.write(decode(m.generate(context, 1000)[0].tolist()))