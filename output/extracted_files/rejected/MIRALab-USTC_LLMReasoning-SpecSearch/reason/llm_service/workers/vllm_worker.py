"""
A model worker that executes the model based on vLLM.

See documentations at docs/vllm_integration.md
"""

import argparse
import asyncio
import json
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from reason.llm_service.workers.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length


app = FastAPI()

def extract_first_logprob(logprobs_list):
    logprobs = []
    if logprobs_list == None:
        return logprobs
    for entry in logprobs_list:
        if entry is None or not entry:  
            continue
        first_key = next(iter(entry))
        first_logprob_info = entry[first_key]
        logprobs.append(first_logprob_info.logprob)
    return logprobs


class VLLMWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        llm_engine: AsyncLLMEngine,
        conv_template: str,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: vLLM worker..."
        )
        self.tokenizer = llm_engine.engine.tokenizer.tokenizer
        self.context_len = get_context_length(llm_engine.engine.model_config.hf_config)

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1 

        context = params.pop("prompt") 
        n = params.get("n", 1)
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1.0)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", True) # False
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)
        include_stop_str_in_output = params.get("include_stop_str_in_output", False)
        seed = params.get("seed", None)


        # Handle stop_str
        stop = set() 
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str) 
        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        flag = 1
        if flag == 1:
            if params.get("max_new_tokens") == 1:
                sampling_params = SamplingParams(
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    # use_beam_search=use_beam_search,
                    stop=list(stop),
                    stop_token_ids=stop_token_ids,
                    max_tokens=max_new_tokens,
                    top_k=top_k,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    best_of=best_of,
                    prompt_logprobs=2,
                    include_stop_str_in_output=include_stop_str_in_output,
                    seed=seed,
                )
                results_generator = engine.generate(context, sampling_params, request_id)
                async for request_output in results_generator: 
                    prompt = request_output.prompt
                    if echo:
                        text_outputs = [
                            prompt + output.text for output in request_output.outputs
                        ]
                    else:
                        text_outputs = [output.text for output in request_output.outputs]
                    # Note: usage is not supported yet
                    prompt_tokens = len(request_output.prompt_token_ids)
                    completion_tokens = sum(
                        len(output.token_ids) for output in request_output.outputs
                    )
                    ret = {
                        "text": text_outputs,
                        "error_code": 0,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                        "cumulative_logprob": [
                            output.cumulative_logprob for output in request_output.outputs
                        ],
                        "prompt_logprob": extract_first_logprob(request_output.prompt_logprobs),
                        "output_token_len": [
                            len(output.token_ids) for output in request_output.outputs
                        ],
                        "finish_reason": (
                            request_output.outputs[0].finish_reason
                            if len(request_output.outputs) == 1
                            else [output.finish_reason for output in request_output.outputs]
                        ),
                    }
                    yield (json.dumps(ret) + "\0").encode()
            else:
                sampling_params = SamplingParams(
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    # use_beam_search=use_beam_search,
                    stop=list(stop),
                    stop_token_ids=stop_token_ids,
                    max_tokens=max_new_tokens,
                    top_k=top_k,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    best_of=best_of,
                    # logprobs=0, # logprob ablation experiment
                    include_stop_str_in_output=include_stop_str_in_output,
                    seed=seed,
                )
                results_generator = engine.generate(context, sampling_params, request_id)
                async for request_output in results_generator: 
                    prompt = request_output.prompt
                    if echo:
                        text_outputs = [
                            prompt + output.text for output in request_output.outputs
                        ]
                    else:
                        text_outputs = [output.text for output in request_output.outputs]
                    # Note: usage is not supported yet
                    prompt_tokens = len(request_output.prompt_token_ids)
                    completion_tokens = sum(
                        len(output.token_ids) for output in request_output.outputs
                    )
                    ret = {
                        "text": text_outputs,
                        "error_code": 0,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                        "cumulative_logprob": [
                            output.cumulative_logprob for output in request_output.outputs
                        ],
                        "output_token_len": [
                            len(output.token_ids) for output in request_output.outputs
                        ],
                        "finish_reason": (
                            request_output.outputs[0].finish_reason
                            if len(request_output.outputs) == 1
                            else [output.finish_reason for output in request_output.outputs]
                        ),
                    }
                    yield (json.dumps(ret) + "\0").encode()
        else:
            sampling_params = SamplingParams(
                n=n,
                temperature=temperature,
                top_p=top_p,
                # use_beam_search=use_beam_search,
                stop=list(stop),
                stop_token_ids=stop_token_ids,
                max_tokens=max_new_tokens,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                best_of=best_of,
                logprobs=0,
                include_stop_str_in_output=include_stop_str_in_output,
                seed=seed,
            )
            results_generator = engine.generate(context, sampling_params, request_id)
            async for request_output in results_generator: 
                logger.info(f"Updated outputs: {request_output}")
                prompt = request_output.prompt
                if echo:
                    text_outputs = [
                        prompt + output.text for output in request_output.outputs
                    ]
                else:
                    text_outputs = [output.text for output in request_output.outputs]
                # Note: usage is not supported yet
                prompt_tokens = len(request_output.prompt_token_ids)
                completion_tokens = sum(
                    len(output.token_ids) for output in request_output.outputs
                )
                ret = {
                    "text": text_outputs,
                    "error_code": 0,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "cumulative_logprob": [
                        output.cumulative_logprob for output in request_output.outputs
                    ],
                    "output_token_len": [
                        len(output.token_ids) for output in request_output.outputs
                    ],
                    "finish_reason": (
                        request_output.outputs[0].finish_reason
                        if len(request_output.outputs) == 1
                        else [output.finish_reason for output in request_output.outputs]
                    ),
                }
                yield (json.dumps(ret) + "\0").encode() 
                
    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode()) 


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        await engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    output = await worker.generate(params)
    release_worker_semaphore()
    await engine.abort(request_id)
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--speculative-model-path", type=str, default=None)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when"
        "downloading the model and tokenizer.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="The ratio (between 0 and 1) of GPU memory to"
        "reserve for the model weights, activations, and KV cache. Higher"
        "values will increase the KV cache size and thus improve the model's"
        "throughput. However, if the value is too high, it may cause out-of-"
        "memory (OOM) errors.",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.model_path:
        args.model = args.model_path
    if args.speculative_model_path:
        args.speculative_model = args.speculative_model_path
        args.num_speculative_tokens = 11
        args.use_v2_block_manager = True
    if args.num_gpus > 1:
        args.tensor_parallel_size = args.num_gpus

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    worker = VLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        engine,
        args.conv_template,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
