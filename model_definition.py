
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import numpy as np
### --------------------- MODEL DEFINITION --------------------- ###


# class Head(nn.Module):
#     def __init__(self, head_size, n_embd):
#         super().__init__()
#         self.query = nn.Linear(n_embd, head_size)
#         self.key = nn.Linear(n_embd, head_size)
#         self.value = nn.Linear(n_embd, head_size)
#         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

#     def forward(self, xb):
#         # xb: (B,T,C), C = n_embd
#         B,T,C = xb.shape
#         q = self.query(xb) # output is (B,T,head_size)
#         k = self.key(xb)


#         # todo: F.scaled_dot_product_attention
#         head_size = q.shape[2]
#         wei = q@k.transpose(-2, -1)*(head_size**-0.5) @ q@k would give us: 
#         wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
#         wei = wei.softmax(dim=-1)

#         v = self.value(xb) # v is B , T, head_size
#         out = wei@v # so out will be B,T, head_size

#         return out


# class MultiAttentionHead(nn.Module):

#     def __init__(self, num_heads, head_size, n_embd):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size=head_size, n_embd=n_embd) for _ in range(num_heads)])
#         self.linear = nn.Linear(num_heads*head_size, n_embd)
#     def forward(self, xb):
#         out = torch.cat([h(xb) for h in self.heads], dim=-1)
#         out = self.linear(out)
#         return out # out is (B,T,n_embd)

class CustomRoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, seq_len: int, n_embd: int):
        super().__init__(seq_len, n_embd) # so now self.weight is of size seq_len x n_embd and initialized. We need to overwrite them
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        seq_len,n_embd = out.shape
        out.requires_grad = False 
        i = torch.arange(1, n_embd/2+1, dtype=float) # so i becomes [1...d/2]
        # we get a user warning here
        thetas = (1/(10000**(2*(i-1)/n_embd))) # thetas are of shape: d/2
        positions = torch.arange(seq_len).unsqueeze(1) # pos of shape seq_len
        m_theta = positions*thetas # shape: seq_len x d/2
        cos =(m_theta.cos()) # seq_len x d/2
        sin =(m_theta.sin()) # seq_len x d/2
        # out = torch.cat([sin, cos], dim=-1) # seq_len x d # incorrect, as out loses the nn.Parameter properties, so we will assign them separately
        out[:, :n_embd//2] = sin
        out[:, n_embd//2:] = cos
        out.detach_()
        return out
    @torch.no_grad()
    def forward(self, seq_len):
        positions = torch.arange(seq_len, device=self.weight.device)
        return super().forward(positions)

    
class CasualSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()

        self.c_attn = nn.Linear(n_embd, 3*n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.num_heads = num_heads
        self.head_size = head_size
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))
        self.embed_positions = CustomRoFormerSinusoidalPositionalEmbedding(seq_len=block_size, n_embd=head_size)

    def apply_rotary_positional_embedding(self, x, sinusoidal_pos):
        # original dimensions of x is: (B,nh,T,d)
        if sinusoidal_pos is None:
            return x
        sin, cos = sinusoidal_pos # sin and cos are of shape: (1,1,T,d/2)
        x1, x2 = x[..., 0::2], x[..., 1::2] # here x1 is say (B,nh,T,d/2)
        # print(x1.shape, x2.shape, sin.shape, cos.shape)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1) # (B,nh,T, d/2 *2) => (B,nh,T,d) what size does this become?


        
    def forward(self,x):
        B,T,n_embd = x.shape
        qkv = self.c_attn(x) # so qkv is of shape [B,T,3*n_embd] 
        q, k, v = qkv.split(n_embd, dim=2) # now we split qkv into q, k, v each of shape [B,T,n_embd] 
        # now the dimension 2 needs to be split based on num_heads
        # (B,nh,T,hs)
        k = k.view(B,T,self.num_heads, self.head_size).transpose(1,2) # so k is now [B,num_heads,T, head_size]
        q = q.view(B,T,self.num_heads, self.head_size).transpose(1,2) # so q is now [B,num_heads,T, head_size]
        v = v.view(B,T,self.num_heads, self.head_size).transpose(1,2) # so v is now [B,num_heads,T, head_size]

        # Apply RoPE positional embedding to q and k
        sinusodial_pos = self.embed_positions(T)[ None, None, :, : ].chunk(2, dim=-1) 
        q = self.apply_rotary_positional_embedding(q, sinusodial_pos)
        k = self.apply_rotary_positional_embedding(k, sinusodial_pos)   

        #need to implement the flash-attention, and also compare the results
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size) # so out is now (B,T,n_embd)
        
        # now we can do the attention
        # wei = q@k.transpose(-2, -1)*(self.head_size**-0.5) # wei is B,nh,T,T
        # wei = wei.masked_fill(self.tril[:,:,:T,:T]==0, float('-inf'))
        # wei = wei.softmax(dim=-1) # wei is (B,nh,T,T)

        # out = wei@v # (B,nh,T,T) @ (B,nh,T,hs) -> B,nh,T,hs
        # out = out.transpose(1,2).contiguous().view(B,T,self.num_heads*self.head_size) # so out is now (B,T,n_embd)
        
        
        out = self.c_proj(out) # so out is now (B,T,n_embd) 
        return out


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4*n_embd) 
        self.gelu = nn.GELU(approximate='tanh')
        self.linear2 = nn.Linear(4*n_embd, n_embd)
        self.linear2.NANOGPT_SCALE_INIT = 1
    
    def forward(self,x):
        out = self.linear1(x)
        out = self.gelu(out)
        out = self.linear2(out)
        return out


class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        self.head_size = n_embd//num_heads
        self.cs_attn = CasualSelfAttention(num_heads=num_heads, head_size=self.head_size, n_embd=n_embd, block_size=block_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x, return_stats=False):
        x = x + self.cs_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# How do i export a class
# we can do this via
class GPT2Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            # wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([Block(n_embd=self.config.n_embd, num_heads=self.config.num_heads, block_size=self.config.block_size) for _ in range(16)]),
            ln_f = nn.LayerNorm(self.config.n_embd)
        ))

        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        # we can use same lm_head and token_emb_table because of weight tying
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        # self.apply(self._print_weight_statistics)


    def _init_weights(self, module):

        std = 0.02
        if isinstance(module, nn.Linear):  
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2*len(self.transformer.h))**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # even the biases are init to zeros
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        

    def forward(self, xb, yb=None):
        B,T = xb.shape
        tok_emb = self.transformer.wte(xb)
        # pos = torch.arange(0, T, device=xb.device)
        # pos_emb = self.transformer.wpe(pos)
        x = tok_emb
        if self.training and self.config.debug_block_stats:
            for block_idx, block in enumerate(self.transformer.h):
                x = block(x)
                print(x.std().item())
            print('========')
        else:
            for block in self.transformer.h:
                x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) # shape of logits is (B,T,vocab_size)

        if(yb is None):
            loss=None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # C is vocab_size
            yb = yb.view(B*T)
            loss = F.cross_entropy(logits, yb)

        return logits, loss

        
# generation part remains same
    def generate(self, xb, max_new_tokens):
        for _ in range(max_new_tokens):
            with torch.no_grad():
                xb_cond = xb[:, -self.config.block_size:]

                logits, _ = self(xb_cond) # here loss is mostly useless

                # let's add temperature here: 
                # by adding temp, we can control the randomness of the output, higher temp means more random, lower temp means more deterministic
                # temperature = 0.8
                # logits = logits / temperature
                logits = logits[:, -1, :] / 0.4 # so logits is now (B,C) where C is vocab size

                topk_probs, topk_indices = torch.topk(logits, k=50, dim=-1) # now we pick the top k tokens, so topk_probs is (B,k) and topk_indices is (B,k)

                probs = F.softmax(topk_probs, dim=-1) # we calculate the probabilities for the last token only, so probs is (B,C)

                ix = torch.multinomial(topk_probs, num_samples=1) # multinomial will sample from the top k probabilities, so ix is (B,1) where the value is the index of the token in the top k
                # so ix is the index of the token in the top k, we need to convert it to the index in the vocab
                ix = topk_indices.gather(-1, ix) # so now ix is (B,1) where the value is the index of the token in the vocab
                # what does gather do? it takes the topk_indices and gathers the values at the indices specified by ix, so we get the actual token index in the vocab
                xb = torch.cat((xb, ix), dim=1) # (B, T+1)

        return xb



### --------------------- --------------------- --------------------- ###
