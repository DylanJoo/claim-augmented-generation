import torch
import json
import re
import os
import string
import time
import gc

def cleanup_vllm(llm):
    del llm.model
    del llm

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.distributed.destroy_process_group()

    return 'vllm engine has been cleaned up'

