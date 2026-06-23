#Set to true if GPU is an Nvidia GPU
NVIDIA_GPU=True
#Set the default device id to use ('cuda', 'cuda:1', 'cuda:2', or 'cpu')
DEFAULT_DEVICE='cuda:1'
#Set to true if it is desired to parallelize training among multiple GPUs
PARALLELIZE_GPUS=False
#Set to true if GPU supports nvidia tensor cores
ENABLE_APEX=False
#set batch size to maximize the memory ussage of GPUs
BATCH_SIZE=1
#usually num_workers=batch_size is fine. Set this value such that Data loading time converges to 0.
NUM_WORKERS=8
