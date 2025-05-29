import gc
import torch

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def init_vllm_llm_vp():
    from vllm import LLM, SamplingParams
    from helper.config import LLM_MODEL_NAME_VP, GPU_MEMORY_UTILIZATION_VP, SAMPLING_PARAMS_VP, BATCH_SIZE_VP
    
    flush_memory()
    device_count = torch.cuda.device_count()
    print(f"Initializing model {LLM_MODEL_NAME_VP} with {device_count} GPU(s)")
    
    sampling_params = SamplingParams(**SAMPLING_PARAMS_VP)
    print(sampling_params)
    llm_instance = LLM(
        model=LLM_MODEL_NAME_VP,
        tensor_parallel_size=device_count,
        trust_remote_code=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION_VP,
        dtype="half",
        max_num_seqs=BATCH_SIZE_VP
    )
    tokenizer = llm_instance.get_tokenizer()
    return llm_instance, tokenizer, sampling_params 

def init_vllm_llm_qg():
    from vllm import LLM, SamplingParams
    from helper.config import LLM_MODEL_NAME_QG, GPU_MEMORY_UTILIZATION_QG, SAMPLING_PARAMS_QG, BATCH_SIZE_QG

    flush_memory()
    device_count = torch.cuda.device_count()
    print(f"Initializing model {LLM_MODEL_NAME_QG} with {device_count} GPU(s)")

    sampling_params = SamplingParams(**SAMPLING_PARAMS_QG)
    print(sampling_params)
    llm_instance = LLM(
        model=LLM_MODEL_NAME_QG,
        tensor_parallel_size=device_count,
        max_model_len=4096,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION_QG,
        enable_prefix_caching=True,
        trust_remote_code=True,
        dtype="bfloat16",
        max_num_seqs=BATCH_SIZE_QG
    )
    tokenizer = llm_instance.get_tokenizer()
    return llm_instance, tokenizer, sampling_params 