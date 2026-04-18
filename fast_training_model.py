### --------------------- LIBRARY IMPORTS --------------------- ###

from pathlib import Path
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import numpy as np

from model_definition import GPT2Model
### ----------------------------------------------------------- ###


### --------------------- DECLARE DEVICE TYPE --------------------- ###
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'  

### --------------------------------------------------------------- ###


### --------------------- DECLARE SEED --------------------- ###

torch.manual_seed(1337)
# torch.backends.cuda.manual_seed_all(1337)
torch.mps.manual_seed(1337)

### -------------------------------------------------------- ###


### --------------------- DECLARE GPT CONFIG --------------------- ###

@dataclass
class GPTConfig:
    block_size: int = 512           # block_size is: sequence size, the number of tokens in a sequence
    batch_size: int = 32            # batch_size is: number of sequences we process in parallel
    n_embd: int = 512               # n_embd is attention blocks dimensions
    learning_rate: float = 3e-4     # learning_rate here is declared as a constant -> might want to update these
    max_grad_norm = 1.0
    num_heads = 8
    debug_block_stats = False

gpt_config = GPTConfig()

### --------------------- --------------------- --------------------- ###




### --------------------- DATA READ AND TOKENIZATION --------------------- ###

tokenizer = tiktoken.get_encoding('gpt2')

data = np.memmap("./dataset/shakespeare.bin", dtype=np.int32, mode="r")
gpt_config.vocab_size = tokenizer.n_vocab

### --------------------- --------------------- --------------------- ###

### --------------------- SIMPLE GET BATCH DATA LOADER --------------------- ###

# here we need to go a little ahead in designing a dataloader class

# so we have to decide on the type of data loader
# for now we are using tiny shakespear, but this will run out quick
# a better choice is: fineweb
# but for that we need to download big shard files


# let's say we continue to use shakespear dataset
# things remain almost the same, but we now need to take in the account of n_gpus, and hence iteration steps

class DataLoaderShakespeare():
    def __init__(self, batch_size, block_size, process_rank, num_processes):

        self.batch_size = batch_size
        self.block_size = block_size

        self.process_rank = process_rank
        self.num_processes = num_processes

        self.position_offset = 0

        self.read_data()

        self.reset_start()

    def read_data(self):
        # dat = np.load("./dataset/shakespeare.bin")
        # dat = dat.astype(np.int32) # added after video
        # self.tokens = torch.tensor(dat, dtype=torch.long)
        dat = np.memmap("./dataset/shakespeare.bin", dtype=np.int32, mode="r")
        self.tokens = torch.tensor(dat, dtype=torch.long)


    def reset_start(self):
        self.current_position = self.batch_size*self.block_size*self.process_rank + self.position_offset # each initialized at [0, 32, 64, 96]

    def get_batch(self, split=None):
        # start_idx of self is now computed
        B, T = self.batch_size, self.block_size

        buf = self.tokens[self.current_position : self.current_position+B*T+1]

        xb = (buf[:-1]).view(B, T) # inputs
        yb = (buf[1:]).view(B, T) # targets

        # now we are going to do checks are re_initialize our current_pos
        self.current_position += B * T * self.num_processes # [0 -> 128, 32->160 and so on]
        # now it is possible that next batch of loading is not possible
        if(self.current_position + (B*T*self.num_processes + 1) > len(self.tokens)): # here we check if we can get the next entire batch or no, otherwise we loop[]
            # here we have a single training data, that we loop over, so, we can include an offset
            if(self.position_offset + B*T*self.num_processes + 1 > len(self.tokens)):
                self.position_offset = 0
            else:
                self.position_offset +=1  # so all the next training loops start at index say 1, there will be a case when offset becomes big enough, that we have to reset it
            # if offset + B*T*num_process + 1 > len(self.tokens): here we reset offset
            self.reset_start()

        return xb, yb




# so get_batch needs to be modified to take in the account of DDP
# we have n_rank, and a world_size
# so we will have to modify the start_idx to take in the account of world_size, so that each process gets a different part of the data
# so for say first iteration, we 
# so let's say B = 4, and T = 8
# so for each GPU, starting point will be: B*T*(n_rank), [0, 32, 64, 96]
# for the next iter: new_pos: start_pos + (B*T*num_proc): 0 + (8*4*4)
# and we also probably want to check if the next batch is possible to load or not
# so let's say: self.current_pos + (B*T*num_proc) + 1 > len(tokens): reset

# OR, we can have a fuzzy and random check
# so for each process, we are only considered if it can fetch the next batch or no
# this only works if, we have a single shard of data, but if we had many, we would have to iterate over them sequentially
# i guess, this is where a dataloader class helps





def get_batch(block_size, batch_size, process_rank,  process_world_size, split=None):
    start_idx = torch.randint(len(data)-block_size-1, (batch_size,))
    # print(start_idx)
    xb = torch.stack([torch.from_numpy(data[idx:idx+block_size].copy()) for idx in start_idx])
    yb = torch.stack([torch.from_numpy(data[idx+1:idx+1+block_size].copy()) for idx in start_idx])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


### --------------------- ---------------------- --------------------- ###



### --------------------- LIBRARIES FOR DDP AND SETUP --------------------- ###


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# set up DDP (distributed data parallel).

# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    
    assert torch.cuda.is_available(), "need CUDA for DDP"

    init_process_group(backend='nccl') # this is required for torch.distributed, CUDA GPUs use 'nccl'
    
    ddp_rank = int(os.environ['RANK']) # RANK gives us the "rank" or "process_number" for the current process seing this code -> helps identify b/w proc

    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # LOCAL_RANK: gives the rank of a process based off num of gpus on a single node, useful for multi-node GPU clusters

    ddp_world_size = int(os.environ['WORLD_SIZE']) # WORLD_SIZE: gives us the total no of process that will be running

    device = f'cuda:{ddp_local_rank}' # since there are many different devices available we need indexes when declaring device: cuda:0, cuda:1 etc

    torch.cuda.set_device(device) # we set the devices

    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc. # here we set a boolean for the master process
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = 'mps'  
    print(f"using device: {device}")

### --------------------- --------------------- --------------------- ###


### --------------------- torch.compile --------------------- ###

model = GPT2Model(config=gpt_config)

model = model.to(device)

# we are only running on cuda for now
model = torch.compile(model) # only possible on cuda for now

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


### --------------------- --------------------- --------------------- ###

optimizer = torch.optim.AdamW(model.parameters(), lr=gpt_config.learning_rate)
steps = 100
# we also want to print the count of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')
# we probably also want to count the number of token the model trains upon during this time, which is B*T*steps
total_tokens = gpt_config.batch_size*gpt_config.block_size*steps*ddp_world_size # we multiply by world size to get the total tokens trained on across all processes
print(f'Total tokens trained on: {total_tokens}')

dataloader = DataLoaderShakespeare(batch_size=gpt_config.batch_size, block_size=gpt_config.block_size, process_rank=ddp_local_rank, num_processes=ddp_world_size)

grad_accum_steps = 8


for step in range(500):
    t0=time.time()
    optimizer.zero_grad()
    loss_accum = 0
    for accum_step in range(grad_accum_steps):

        xb, yb = dataloader.get_batch()

        xb, yb = xb.to(device), yb.to(device)
        
        B,T = xb.shape
        
        logits, loss = model(xb, yb)
            
        # import code; code.interact(local=locals())
        loss = loss/grad_accum_steps

        loss_accum += loss.item()

        loss.backward()


    if device == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    elif device == "mps":
        torch.mps.synchronize()

    t1=time.time()
    if(step%10 == 0):
        dt = (t1-t0)*1000
        print(f'token throughput: {B*T/dt} tokens/ms, Loss: {loss_accum}')
    
    # we also want to do gradient clipping here, which we will do via norm grad clip:
    torch.nn.utils.clip_grad_norm_(model.parameters(), gpt_config.max_grad_norm)
    optimizer.step()

# generation part remains same
start_text = "I am a language model, "
encoded_text = tokenizer.encode(start_text)
start_sequence = torch.tensor(encoded_text, dtype=torch.long).to(device=device)
start_sequence = start_sequence.unsqueeze(0) # add batch dimension

generated_tensor = model.generate(start_sequence, 60)
tensor_values = generated_tensor[0].tolist()

# # we need to map these token id based on vocab
# # decode function takes raw tensor values
print(tokenizer.decode(tensor_values))



if ddp:
    destroy_process_group()