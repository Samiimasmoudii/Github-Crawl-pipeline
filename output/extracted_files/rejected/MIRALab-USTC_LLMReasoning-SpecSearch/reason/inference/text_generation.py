from typing import List, Optional
import requests
from dataclasses import dataclass
import time
import logging


@dataclass
class ConcatedLMGenResult:
    text: List[str]
    prompt_tokens: List[int]
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]
    prompt_logprob: float
    finish_reason: List[str]
    total_time: float
    # logprob: List[str]

    # post init compute number of completion_tokens
    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)

@dataclass
class BlockLMGenResult:
    text: List[str]
    prompt_tokens: List[int]
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]
    prompt_logprob: List
    logprob: List
    finish_reason: List[str]
    total_time: float

    # post init compute number of completion_tokens
    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)

# Generate text through remote calls, first query the model's worker address, and then send a request to that address to generate text
def _generate_fastchat( 
    query_str, 
    model_name,
    n, 
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    stop_token_ids, 
    stop_str,
    include_stop_str_in_output,
    seed,
    controller_addr,
) -> ConcatedLMGenResult:

    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    if not worker_addr:
        raise ValueError("Language Model name {} does not exist.".format(model_name))

    headers = {"User-Agent": "FastChat Client"} 
    gen_params = { 
        "model": model_name,
        "prompt": query_str,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "top_k": top_k,
        "stop_token_ids": stop_token_ids,
        "max_new_tokens": max_new_tokens,
        "stop": stop_str,
        "echo": False,
        "include_stop_str_in_output": include_stop_str_in_output,
        "seed": seed,
    }
    start_time = time.time()
    # Use requests.post to send a POST request with generated parameters to the specified remote service address
    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    total_time = time.time() - start_time
    results = response.json() 
    output_token_lens = results["output_token_len"]
    cum_logps = results["cumulative_logprob"] 
    avg_len_logps = [0]
    flag = 1
    if flag == 1:
        if max_new_tokens == 1:
            return ConcatedLMGenResult(
                text=results["text"],
                prompt_tokens=results["usage"]["prompt_tokens"],
                prompt_logprob= results["prompt_logprob"],
                num_tokens=results["output_token_len"],
                cumulative_logprob=results["cumulative_logprob"],
                logp_avg_by_len=avg_len_logps,
                finish_reason=results["finish_reason"],
                total_time=total_time,
            )
        else:
            return ConcatedLMGenResult(
                text=results["text"],
                prompt_tokens=results["usage"]["prompt_tokens"],
                prompt_logprob= [],
                num_tokens=results["output_token_len"],
                cumulative_logprob=results["cumulative_logprob"],
                logp_avg_by_len=avg_len_logps,
                finish_reason=results["finish_reason"],
                total_time=total_time,
            )
    else:
        return ConcatedLMGenResult(
            text=results["text"],
            prompt_tokens=results["usage"]["prompt_tokens"],
            prompt_logprob=[],
            num_tokens=results["output_token_len"],
            cumulative_logprob=results["cumulative_logprob"],
            logp_avg_by_len=avg_len_logps,
            finish_reason=results["finish_reason"],
            total_time=total_time,
            logprob=results["logprob"],
        )

def _generate_speculative_fastchat(
    query_str, #
    model_name,
    speculative_model_name,
    n, 
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    stop_token_ids, 
    stop_str,
    include_stop_str_in_output,
    seed,
    controller_addr,
) -> ConcatedLMGenResult:
    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    if not worker_addr:
        raise ValueError("Language Model name {} {} does not exist.".format(model_name,speculative_model_name))

    headers = {"User-Agent": "FastChat Client"} 
    gen_params = { 
        "model": model_name,
        "speculative_model": speculative_model_name,
        "prompt": query_str,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "top_k": top_k,
        "stop_token_ids": stop_token_ids,
        "max_new_tokens": max_new_tokens,
        "stop": stop_str,
        "echo": False,
        "include_stop_str_in_output": include_stop_str_in_output,
        "seed": seed,
    }
    start_time = time.time()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    total_time = time.time() - start_time
    results = response.json() 
    output_token_lens = results["output_token_len"]
    cum_logps = results["cumulative_logprob"] 
    avg_len_logps = [0]
    flag = 1
    if flag == 1:
        if max_new_tokens == 1:
            return ConcatedLMGenResult(
                text=results["text"],
                prompt_tokens=results["usage"]["prompt_tokens"],
                prompt_logprob= results["prompt_logprob"],
                num_tokens=results["output_token_len"],
                cumulative_logprob=results["cumulative_logprob"],
                logp_avg_by_len=avg_len_logps,
                finish_reason=results["finish_reason"],
                total_time=total_time,
            )
        else:
            return ConcatedLMGenResult(
                text=results["text"],
                prompt_tokens=results["usage"]["prompt_tokens"],
                prompt_logprob= [],
                num_tokens=results["output_token_len"],
                cumulative_logprob=results["cumulative_logprob"],
                logp_avg_by_len=avg_len_logps,
                finish_reason=results["finish_reason"],
                total_time=total_time,
            )
    else:
        return ConcatedLMGenResult(
            text=results["text"],
            prompt_tokens=results["usage"]["prompt_tokens"],
            prompt_logprob=[],
            num_tokens=results["output_token_len"],
            cumulative_logprob=results["cumulative_logprob"],
            logp_avg_by_len=avg_len_logps,
            finish_reason=results["finish_reason"],
            total_time=total_time,
            logprob=results["logprob"],
        )