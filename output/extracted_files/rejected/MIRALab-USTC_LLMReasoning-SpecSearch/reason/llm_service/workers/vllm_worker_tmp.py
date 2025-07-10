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
        if entry is None or not entry:  # 检查条目是否为 None 或空字典
            continue
        # 获取字典中的第一个键值对
        first_key = next(iter(entry))
        first_logprob_info = entry[first_key]
        logprobs.append(first_logprob_info.logprob)
    return logprobs

def extract_first_logprob_token(logprobs_list):
    logprobs = []
    token = []
    for entry in logprobs_list:
        if entry is None or not entry:  # 检查条目是否为 None 或空字典
            continue
        # 获取字典中的第一个键值对
        first_key = next(iter(entry))
        first_logprob_info = entry[first_key]
        logprobs.append(first_logprob_info.logprob)
        token.append(first_logprob_info.decoded_token)
    return logprobs, token

def extract_first_logprob_and_two_tokens(logprobs_list):
    first_logprobs = []
    first_tokens = []
    second_tokens = []

    if logprobs_list == None:
        return first_logprobs, first_tokens, second_tokens

    for entry in logprobs_list:
        if entry is None or not entry:  # 检查条目是否为 None 或空字典
            continue

        items = list(entry.items())  # 将字典项转换为列表以便索引访问

        # 提取第一个元素的 logprob 和 decoded_token
        first_key, first_logprob_info = items[0] if items else (None, None)
        if first_logprob_info is not None:
            first_logprobs.append(first_logprob_info.logprob)
            first_tokens.append(first_logprob_info.decoded_token)

        # 提取第二个元素的 decoded_token，如果没有则添加空字符串
        second_item = items[1] if len(items) > 1 else None
        if second_item:
            _, second_logprob_info = second_item
            second_tokens.append(second_logprob_info.decoded_token)
        else:
            second_tokens.append("")

    return first_logprobs, first_tokens, second_tokens


def extract_decoded_token_and_logprob(prompt_logprobs):
    logprobs = []
    if prompt_logprobs == None:
        return logprobs
    for entry in prompt_logprobs:
        if entry is None:
            continue
        for token_id, logprob_info in entry.items():
            logprobs.append(logprob_info.logprob)
    return logprobs

def extract_logprobs(logprobs_list):
    result = []
    if logprobs_list == None:
        return result
    for item in logprobs_list:
        if item is not None:
            inner_result = []
            for token_id, logprob_obj in item.items():
                inner_result.append({
                    'decoded_token': logprob_obj.decoded_token,
                    'logprob': logprob_obj.logprob
                })
            result.append(inner_result)
    return result

# def sum_logprobs_after_double_newline(prompt_logprobs):
#     total_sum = 0.0
#     found_double_newline = False

#     # 从倒数第二个元素开始从后往前遍历 prompt_logprobs
#     for logprobs_dict in reversed(prompt_logprobs[:-1]):
#         if logprobs_dict is None:
#             continue

#         # 只考虑字典中的第一个条目
#         first_item = next(iter(logprobs_dict.items()))
#         token_id, logprob = first_item

#         if logprob.decoded_token == '.\n\n':
#             found_double_newline = True
#             break

#         total_sum += logprob.logprob

#     # 加上最后一个元素的 logprob 值
#     if prompt_logprobs[-1] is not None:
#         last_item = next(iter(prompt_logprobs[-1].items()))
#         _, last_logprob = last_item
#         total_sum += last_logprob.logprob

#     return total_sum

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

    # 一个异步生成器方法，接收一个参数 params，包含生成文本所需的各项参数
    async def generate_stream(self, params):
        self.call_ct += 1 # 记录调用次数

        context = params.pop("prompt") # 从 params 中提取 prompt 参数，并将其从 params 中移除
        n = params.get("n", 1) # 从 params 中提取 n 参数，表示生成的文本数量，默认值为 1
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
        # seed = 151

        # Handle stop_str
        stop = set() # 创建一个空集合 stop，用于存储停止字符串
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str) # 如果 stop_str 是非空列表，则将列表中所有字符串添加到stop集合中

        # for tid in stop_token_ids:
        #     if tid is not None:
        #         stop.add(self.tokenizer.decode(tid))

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0

        flag = 1
        # if params.get("model") == 'Qwen2.5-Math-7B-Instruct':
        if flag == 1:
            # # token 
            # if params.get("max_new_tokens") == 1 and params.get("model") == 'Qwen2.5-Math-7B-Instruct':
            #     sampling_params = SamplingParams(
            #         n=n,
            #         temperature=temperature,
            #         top_p=top_p,
            #         use_beam_search=use_beam_search,
            #         stop=list(stop),
            #         stop_token_ids=stop_token_ids,
            #         max_tokens=max_new_tokens,
            #         top_k=top_k,
            #         presence_penalty=presence_penalty,
            #         frequency_penalty=frequency_penalty,
            #         best_of=best_of,
            #         prompt_logprobs=2,
            #         include_stop_str_in_output=include_stop_str_in_output,
            #         seed=seed,
            #     )
            #     results_generator = engine.generate(context, sampling_params, request_id)
            #     async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
            #         # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
            #         # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
            #         prompt = request_output.prompt
            #         if echo:
            #             text_outputs = [
            #                 prompt + output.text for output in request_output.outputs
            #             ]
            #         else:
            #             text_outputs = [output.text for output in request_output.outputs]
            #         # text_outputs = " ".join(text_outputs)
            #         # Note: usage is not supported yet
            #         prompt_tokens = len(request_output.prompt_token_ids)
            #         completion_tokens = sum(
            #             len(output.token_ids) for output in request_output.outputs
            #         )
            #         # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
            #         logprob, token, token2 = extract_first_logprob_and_two_tokens(request_output.prompt_logprobs)
            #         ret = {
            #             "text": text_outputs,
            #             "error_code": 0,
            #             "usage": {
            #                 "prompt_tokens": prompt_tokens,
            #                 "completion_tokens": completion_tokens,
            #                 "total_tokens": prompt_tokens + completion_tokens,
            #             },
            #             "cumulative_logprob": [
            #                 output.cumulative_logprob for output in request_output.outputs
            #             ],
            #             "prompt_logprob": [logprob, token, token2],
            #             "output_token_len": [
            #                 len(output.token_ids) for output in request_output.outputs
            #             ],
            #             "finish_reason": (
            #                 request_output.outputs[0].finish_reason
            #                 if len(request_output.outputs) == 1
            #                 else [output.finish_reason for output in request_output.outputs]
            #             ),
            #         }
            #         # logger.info(f"Updated ret: {ret}")
            #         yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔
            # elif params.get("model") == 'Qwen2.5-Math-1.5B-Instruct':
            #     sampling_params = SamplingParams(
            #         n=n,
            #         temperature=temperature,
            #         top_p=top_p,
            #         use_beam_search=use_beam_search,
            #         stop=list(stop),
            #         stop_token_ids=stop_token_ids,
            #         max_tokens=max_new_tokens,
            #         top_k=top_k,
            #         presence_penalty=presence_penalty,
            #         frequency_penalty=frequency_penalty,
            #         best_of=best_of,
            #         logprobs=0,
            #         include_stop_str_in_output=include_stop_str_in_output,
            #         seed=seed,
            #     )
            #     results_generator = engine.generate(context, sampling_params, request_id)
            #     async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
            #         # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
            #         # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
            #         prompt = request_output.prompt
            #         if echo:
            #             text_outputs = [
            #                 prompt + output.text for output in request_output.outputs
            #             ]
            #         else:
            #             text_outputs = [output.text for output in request_output.outputs]
            #         # text_outputs = " ".join(text_outputs)
            #         # Note: usage is not supported yet
            #         prompt_tokens = len(request_output.prompt_token_ids)
            #         completion_tokens = sum(
            #             len(output.token_ids) for output in request_output.outputs
            #         )
            #         # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
            #         ret = {
            #             "text": text_outputs,
            #             "error_code": 0,
            #             "usage": {
            #                 "prompt_tokens": prompt_tokens,
            #                 "completion_tokens": completion_tokens,
            #                 "total_tokens": prompt_tokens + completion_tokens,
            #             },
            #             "cumulative_logprob": [
            #                 output.cumulative_logprob for output in request_output.outputs
            #             ],
            #             "logprob": extract_first_logprob(request_output.outputs[0].logprobs),
            #             "output_token_len": [
            #                 len(output.token_ids) for output in request_output.outputs
            #             ],
            #             "finish_reason": (
            #                 request_output.outputs[0].finish_reason
            #                 if len(request_output.outputs) == 1
            #                 else [output.finish_reason for output in request_output.outputs]
            #             ),
            #         }
            #         # logger.info(f"Updated ret: {ret}")
            #         yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔
            # else:
            #     sampling_params = SamplingParams(
            #         n=n,
            #         temperature=temperature,
            #         top_p=top_p,
            #         use_beam_search=use_beam_search,
            #         stop=list(stop),
            #         stop_token_ids=stop_token_ids,
            #         max_tokens=max_new_tokens,
            #         top_k=top_k,
            #         presence_penalty=presence_penalty,
            #         frequency_penalty=frequency_penalty,
            #         best_of=best_of,
            #         include_stop_str_in_output=include_stop_str_in_output,
            #         seed=seed,
            #     )
            #     results_generator = engine.generate(context, sampling_params, request_id)
            #     async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
            #         # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
            #         # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
            #         # logger.info(f"Updated request_output: {request_output}")
            #         prompt = request_output.prompt
            #         if echo:
            #             text_outputs = [
            #                 prompt + output.text for output in request_output.outputs
            #             ]
            #         else:
            #             text_outputs = [output.text for output in request_output.outputs]
            #         # text_outputs = " ".join(text_outputs)
            #         # Note: usage is not supported yet
            #         prompt_tokens = len(request_output.prompt_token_ids)
            #         completion_tokens = sum(
            #             len(output.token_ids) for output in request_output.outputs
            #         )
            #         # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
            #         ret = {
            #             "text": text_outputs,
            #             "error_code": 0,
            #             "usage": {
            #                 "prompt_tokens": prompt_tokens,
            #                 "completion_tokens": completion_tokens,
            #                 "total_tokens": prompt_tokens + completion_tokens,
            #             },
            #             "cumulative_logprob": [
            #                 output.cumulative_logprob for output in request_output.outputs
            #             ],
            #             "output_token_len": [
            #                 len(output.token_ids) for output in request_output.outputs
            #             ],
            #             "finish_reason": (
            #                 request_output.outputs[0].finish_reason
            #                 if len(request_output.outputs) == 1
            #                 else [output.finish_reason for output in request_output.outputs]
            #             ),
            #         }
            #         # logger.info(f"Updated ret: {ret}")
            #         yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔

            # block v3
            if params.get("max_new_tokens") == 1:
                sampling_params = SamplingParams(
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    use_beam_search=use_beam_search,
                    stop=list(stop),
                    stop_token_ids=stop_token_ids,
                    max_tokens=max_new_tokens,
                    top_k=top_k,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    best_of=best_of,
                    prompt_logprobs=0,
                    include_stop_str_in_output=include_stop_str_in_output,
                    seed=seed,
                )
                results_generator = engine.generate(context, sampling_params, request_id)
                async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
                    # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
                    # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
                    prompt = request_output.prompt
                    if echo:
                        text_outputs = [
                            prompt + output.text for output in request_output.outputs
                        ]
                    else:
                        text_outputs = [output.text for output in request_output.outputs]
                    # text_outputs = " ".join(text_outputs)
                    # Note: usage is not supported yet
                    prompt_tokens = len(request_output.prompt_token_ids)
                    completion_tokens = sum(
                        len(output.token_ids) for output in request_output.outputs
                    )
                    # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
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
                    # logger.info(f"Updated ret: {ret}")
                    yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔
            # elif params.get("max_new_tokens") == 1:
            #     sampling_params = SamplingParams(
            #         n=n,
            #         temperature=temperature,
            #         top_p=top_p,
            #         use_beam_search=use_beam_search,
            #         stop=list(stop),
            #         stop_token_ids=stop_token_ids,
            #         max_tokens=max_new_tokens,
            #         top_k=top_k,
            #         presence_penalty=presence_penalty,
            #         frequency_penalty=frequency_penalty,
            #         best_of=best_of,
            #         logprobs=20,
            #         include_stop_str_in_output=include_stop_str_in_output,
            #         seed=seed,
            #     )
            #     results_generator = engine.generate(context, sampling_params, request_id)
            #     async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
            #         # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
            #         # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
            #         prompt = request_output.prompt
            #         if echo:
            #             text_outputs = [
            #                 prompt + output.text for output in request_output.outputs
            #             ]
            #         else:
            #             text_outputs = [output.text for output in request_output.outputs]
            #         # text_outputs = " ".join(text_outputs)
            #         # Note: usage is not supported yet
            #         prompt_tokens = len(request_output.prompt_token_ids)
            #         completion_tokens = sum(
            #             len(output.token_ids) for output in request_output.outputs
            #         )
            #         # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
            #         ret = {
            #             "text": text_outputs,
            #             "error_code": 0,
            #             "usage": {
            #                 "prompt_tokens": prompt_tokens,
            #                 "completion_tokens": completion_tokens,
            #                 "total_tokens": prompt_tokens + completion_tokens,
            #             },
            #             "cumulative_logprob": [
            #                 output.cumulative_logprob for output in request_output.outputs
            #             ],
            #             "logprob": extract_logprobs(request_output.outputs[0].logprobs),
            #             "output_token_len": [
            #                 len(output.token_ids) for output in request_output.outputs
            #             ],
            #             "finish_reason": (
            #                 request_output.outputs[0].finish_reason
            #                 if len(request_output.outputs) == 1
            #                 else [output.finish_reason for output in request_output.outputs]
            #             ),
            #         }
            #         # logger.info(f"Updated ret: {ret}")
            #         yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔
            elif params.get("model") == 'Qwen2.5-Math-1.5B-Instruct':
                sampling_params = SamplingParams(
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    use_beam_search=use_beam_search,
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
                async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
                    # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
                    # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
                    # logger.info(f"Updated result: {request_output.outputs}")
                    prompt = request_output.prompt
                    if echo:
                        text_outputs = [
                            prompt + output.text for output in request_output.outputs
                        ]
                    else:
                        text_outputs = [output.text for output in request_output.outputs]
                    # text_outputs = " ".join(text_outputs)
                    # Note: usage is not supported yet
                    prompt_tokens = len(request_output.prompt_token_ids)
                    completion_tokens = sum(
                        len(output.token_ids) for output in request_output.outputs
                    )
                    if len(text_outputs) > 1:
                        logprob = []
                        token = []
                        for output in request_output.outputs:
                            logprob_, token_ = extract_first_logprob_token(output.logprobs)
                            logprob.append(logprob_)
                            token.append(token_)

                    else:
                        logprob, token = extract_first_logprob_token(request_output.outputs[0].logprobs)
                    # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
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
                        "logprob": logprob,
                        "text_token": token,
                        "output_token_len": [
                            len(output.token_ids) for output in request_output.outputs
                        ],
                        "finish_reason": (
                            request_output.outputs[0].finish_reason
                            if len(request_output.outputs) == 1
                            else [output.finish_reason for output in request_output.outputs]
                        ),
                    }
                    # logger.info(f"Updated ret: {ret}")
                    yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔
            else:
                sampling_params = SamplingParams(
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    use_beam_search=use_beam_search,
                    stop=list(stop),
                    stop_token_ids=stop_token_ids,
                    max_tokens=max_new_tokens,
                    top_k=top_k,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    best_of=best_of,
                    include_stop_str_in_output=include_stop_str_in_output,
                    seed=seed,
                )
                results_generator = engine.generate(context, sampling_params, request_id)
                async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
                    # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
                    # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
                    # logger.info(f"Updated request_output: {request_output}")
                    prompt = request_output.prompt
                    if echo:
                        text_outputs = [
                            prompt + output.text for output in request_output.outputs
                        ]
                    else:
                        text_outputs = [output.text for output in request_output.outputs]
                    # text_outputs = " ".join(text_outputs)
                    # Note: usage is not supported yet
                    prompt_tokens = len(request_output.prompt_token_ids)
                    completion_tokens = sum(
                        len(output.token_ids) for output in request_output.outputs
                    )
                    # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
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
                    # logger.info(f"Updated ret: {ret}")
                    yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔

            # # block v2
            # if params.get("max_new_tokens") == 1 and params.get("model") == 'Qwen2.5-Math-7B-Instruct':
            #     sampling_params = SamplingParams(
            #         n=n,
            #         temperature=temperature,
            #         top_p=top_p,
            #         use_beam_search=use_beam_search,
            #         stop=list(stop),
            #         stop_token_ids=stop_token_ids,
            #         max_tokens=max_new_tokens,
            #         top_k=top_k,
            #         presence_penalty=presence_penalty,
            #         frequency_penalty=frequency_penalty,
            #         best_of=best_of,
            #         prompt_logprobs=0,
            #         include_stop_str_in_output=include_stop_str_in_output,
            #         seed=seed,
            #     )
            #     results_generator = engine.generate(context, sampling_params, request_id)
            #     async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
            #         # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
            #         # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
            #         prompt = request_output.prompt
            #         if echo:
            #             text_outputs = [
            #                 prompt + output.text for output in request_output.outputs
            #             ]
            #         else:
            #             text_outputs = [output.text for output in request_output.outputs]
            #         # text_outputs = " ".join(text_outputs)
            #         # Note: usage is not supported yet
            #         prompt_tokens = len(request_output.prompt_token_ids)
            #         completion_tokens = sum(
            #             len(output.token_ids) for output in request_output.outputs
            #         )
            #         # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
            #         ret = {
            #             "text": text_outputs,
            #             "error_code": 0,
            #             "usage": {
            #                 "prompt_tokens": prompt_tokens,
            #                 "completion_tokens": completion_tokens,
            #                 "total_tokens": prompt_tokens + completion_tokens,
            #             },
            #             "cumulative_logprob": [
            #                 output.cumulative_logprob for output in request_output.outputs
            #             ],
            #             "prompt_logprob": extract_first_logprob(request_output.prompt_logprobs),
            #             "output_token_len": [
            #                 len(output.token_ids) for output in request_output.outputs
            #             ],
            #             "finish_reason": (
            #                 request_output.outputs[0].finish_reason
            #                 if len(request_output.outputs) == 1
            #                 else [output.finish_reason for output in request_output.outputs]
            #             ),
            #         }
            #         # logger.info(f"Updated ret: {ret}")
            #         yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔
            # elif params.get("model") == 'Qwen2.5-Math-1.5B-Instruct':
            #     sampling_params = SamplingParams(
            #         n=n,
            #         temperature=temperature,
            #         top_p=top_p,
            #         use_beam_search=use_beam_search,
            #         stop=list(stop),
            #         stop_token_ids=stop_token_ids,
            #         max_tokens=max_new_tokens,
            #         top_k=top_k,
            #         presence_penalty=presence_penalty,
            #         frequency_penalty=frequency_penalty,
            #         best_of=best_of,
            #         logprobs=0,
            #         include_stop_str_in_output=include_stop_str_in_output,
            #         seed=seed,
            #     )
            #     results_generator = engine.generate(context, sampling_params, request_id)
            #     async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
            #         # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
            #         # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
            #         prompt = request_output.prompt
            #         if echo:
            #             text_outputs = [
            #                 prompt + output.text for output in request_output.outputs
            #             ]
            #         else:
            #             text_outputs = [output.text for output in request_output.outputs]
            #         # text_outputs = " ".join(text_outputs)
            #         # Note: usage is not supported yet
            #         prompt_tokens = len(request_output.prompt_token_ids)
            #         completion_tokens = sum(
            #             len(output.token_ids) for output in request_output.outputs
            #         )
            #         logprob, token = extract_first_logprob_token(request_output.outputs[0].logprobs)
            #         # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
            #         ret = {
            #             "text": text_outputs,
            #             "error_code": 0,
            #             "usage": {
            #                 "prompt_tokens": prompt_tokens,
            #                 "completion_tokens": completion_tokens,
            #                 "total_tokens": prompt_tokens + completion_tokens,
            #             },
            #             "cumulative_logprob": [
            #                 output.cumulative_logprob for output in request_output.outputs
            #             ],
            #             "logprob": logprob,
            #             "text_token": token,
            #             "output_token_len": [
            #                 len(output.token_ids) for output in request_output.outputs
            #             ],
            #             "finish_reason": (
            #                 request_output.outputs[0].finish_reason
            #                 if len(request_output.outputs) == 1
            #                 else [output.finish_reason for output in request_output.outputs]
            #             ),
            #         }
            #         # logger.info(f"Updated ret: {ret}")
            #         yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔
            # else:
            #     sampling_params = SamplingParams(
            #         n=n,
            #         temperature=temperature,
            #         top_p=top_p,
            #         use_beam_search=use_beam_search,
            #         stop=list(stop),
            #         stop_token_ids=stop_token_ids,
            #         max_tokens=max_new_tokens,
            #         top_k=top_k,
            #         presence_penalty=presence_penalty,
            #         frequency_penalty=frequency_penalty,
            #         best_of=best_of,
            #         include_stop_str_in_output=include_stop_str_in_output,
            #         seed=seed,
            #     )
            #     results_generator = engine.generate(context, sampling_params, request_id)
            #     async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
            #         # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
            #         # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
            #         # logger.info(f"Updated request_output: {request_output}")
            #         prompt = request_output.prompt
            #         if echo:
            #             text_outputs = [
            #                 prompt + output.text for output in request_output.outputs
            #             ]
            #         else:
            #             text_outputs = [output.text for output in request_output.outputs]
            #         # text_outputs = " ".join(text_outputs)
            #         # Note: usage is not supported yet
            #         prompt_tokens = len(request_output.prompt_token_ids)
            #         completion_tokens = sum(
            #             len(output.token_ids) for output in request_output.outputs
            #         )
            #         # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
            #         ret = {
            #             "text": text_outputs,
            #             "error_code": 0,
            #             "usage": {
            #                 "prompt_tokens": prompt_tokens,
            #                 "completion_tokens": completion_tokens,
            #                 "total_tokens": prompt_tokens + completion_tokens,
            #             },
            #             "cumulative_logprob": [
            #                 output.cumulative_logprob for output in request_output.outputs
            #             ],
            #             "output_token_len": [
            #                 len(output.token_ids) for output in request_output.outputs
            #             ],
            #             "finish_reason": (
            #                 request_output.outputs[0].finish_reason
            #                 if len(request_output.outputs) == 1
            #                 else [output.finish_reason for output in request_output.outputs]
            #             ),
            #         }
            #         # logger.info(f"Updated ret: {ret}")
            #         yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔

            # # block v1
            # if params.get("max_new_tokens") == 1 and params.get("model") == 'Qwen2.5-Math-7B-Instruct':
            #     sampling_params = SamplingParams(
            #         n=n,
            #         temperature=temperature,
            #         top_p=top_p,
            #         use_beam_search=use_beam_search,
            #         stop=list(stop),
            #         stop_token_ids=stop_token_ids,
            #         max_tokens=max_new_tokens,
            #         top_k=top_k,
            #         presence_penalty=presence_penalty,
            #         frequency_penalty=frequency_penalty,
            #         best_of=best_of,
            #         prompt_logprobs=20,
            #         include_stop_str_in_output=include_stop_str_in_output,
            #         seed=seed,
            #     )
            #     results_generator = engine.generate(context, sampling_params, request_id)
            #     async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
            #         logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
            #         # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
            #         prompt = request_output.prompt
            #         if echo:
            #             text_outputs = [
            #                 prompt + output.text for output in request_output.outputs
            #             ]
            #         else:
            #             text_outputs = [output.text for output in request_output.outputs]
            #         # text_outputs = " ".join(text_outputs)
            #         # Note: usage is not supported yet
            #         prompt_tokens = len(request_output.prompt_token_ids)
            #         completion_tokens = sum(
            #             len(output.token_ids) for output in request_output.outputs
            #         )
            #         # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
            #         ret = {
            #             "text": text_outputs,
            #             "error_code": 0,
            #             "usage": {
            #                 "prompt_tokens": prompt_tokens,
            #                 "completion_tokens": completion_tokens,
            #                 "total_tokens": prompt_tokens + completion_tokens,
            #             },
            #             "cumulative_logprob": [
            #                 output.cumulative_logprob for output in request_output.outputs
            #             ],
            #             "prompt_logprob": extract_logprobs(request_output.prompt_logprobs),
            #             "output_token_len": [
            #                 len(output.token_ids) for output in request_output.outputs
            #             ],
            #             "finish_reason": (
            #                 request_output.outputs[0].finish_reason
            #                 if len(request_output.outputs) == 1
            #                 else [output.finish_reason for output in request_output.outputs]
            #             ),
            #         }
            #         # logger.info(f"Updated ret: {ret}")
            #         yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔
            # elif params.get("model") == 'Qwen2.5-Math-1.5B-Instruct':
            #     sampling_params = SamplingParams(
            #         n=n,
            #         temperature=temperature,
            #         top_p=top_p,
            #         use_beam_search=use_beam_search,
            #         stop=list(stop),
            #         stop_token_ids=stop_token_ids,
            #         max_tokens=max_new_tokens,
            #         top_k=top_k,
            #         presence_penalty=presence_penalty,
            #         frequency_penalty=frequency_penalty,
            #         best_of=best_of,
            #         logprobs=20,
            #         include_stop_str_in_output=include_stop_str_in_output,
            #         seed=seed,
            #     )
            #     results_generator = engine.generate(context, sampling_params, request_id)
            #     async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
            #         # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
            #         logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
            #         prompt = request_output.prompt
            #         if echo:
            #             text_outputs = [
            #                 prompt + output.text for output in request_output.outputs
            #             ]
            #         else:
            #             text_outputs = [output.text for output in request_output.outputs]
            #         # text_outputs = " ".join(text_outputs)
            #         # Note: usage is not supported yet
            #         prompt_tokens = len(request_output.prompt_token_ids)
            #         completion_tokens = sum(
            #             len(output.token_ids) for output in request_output.outputs
            #         )
            #         # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
            #         ret = {
            #             "text": text_outputs,
            #             "error_code": 0,
            #             "usage": {
            #                 "prompt_tokens": prompt_tokens,
            #                 "completion_tokens": completion_tokens,
            #                 "total_tokens": prompt_tokens + completion_tokens,
            #             },
            #             "cumulative_logprob": [
            #                 output.cumulative_logprob for output in request_output.outputs
            #             ],
            #             "logprob": extract_logprobs(request_output.outputs[0].logprobs),
            #             "output_token_len": [
            #                 len(output.token_ids) for output in request_output.outputs
            #             ],
            #             "finish_reason": (
            #                 request_output.outputs[0].finish_reason
            #                 if len(request_output.outputs) == 1
            #                 else [output.finish_reason for output in request_output.outputs]
            #             ),
            #         }
            #         # logger.info(f"Updated ret: {ret}")
            #         yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔
            # else:
            #     sampling_params = SamplingParams(
            #         n=n,
            #         temperature=temperature,
            #         top_p=top_p,
            #         use_beam_search=use_beam_search,
            #         stop=list(stop),
            #         stop_token_ids=stop_token_ids,
            #         max_tokens=max_new_tokens,
            #         top_k=top_k,
            #         presence_penalty=presence_penalty,
            #         frequency_penalty=frequency_penalty,
            #         best_of=best_of,
            #         include_stop_str_in_output=include_stop_str_in_output,
            #         seed=seed,
            #     )
            #     results_generator = engine.generate(context, sampling_params, request_id)
            #     async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
            #         # logger.info(f"Updated prompt_logprobs: {request_output.prompt_logprobs}")
            #         # logger.info(f"Updated logprobs: {request_output.outputs[0].logprobs}")
            #         logger.info(f"Updated request_output: {request_output}")
            #         prompt = request_output.prompt
            #         if echo:
            #             text_outputs = [
            #                 prompt + output.text for output in request_output.outputs
            #             ]
            #         else:
            #             text_outputs = [output.text for output in request_output.outputs]
            #         # text_outputs = " ".join(text_outputs)
            #         # Note: usage is not supported yet
            #         prompt_tokens = len(request_output.prompt_token_ids)
            #         completion_tokens = sum(
            #             len(output.token_ids) for output in request_output.outputs
            #         )
            #         # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
            #         ret = {
            #             "text": text_outputs,
            #             "error_code": 0,
            #             "usage": {
            #                 "prompt_tokens": prompt_tokens,
            #                 "completion_tokens": completion_tokens,
            #                 "total_tokens": prompt_tokens + completion_tokens,
            #             },
            #             "cumulative_logprob": [
            #                 output.cumulative_logprob for output in request_output.outputs
            #             ],
            #             "output_token_len": [
            #                 len(output.token_ids) for output in request_output.outputs
            #             ],
            #             "finish_reason": (
            #                 request_output.outputs[0].finish_reason
            #                 if len(request_output.outputs) == 1
            #                 else [output.finish_reason for output in request_output.outputs]
            #             ),
            #         }
            #         # logger.info(f"Updated ret: {ret}")
            #         yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔


        elif params.get("max_new_tokens") == 1 and params.get("model") == 'Qwen2.5-Math-7B-Instruct':
            sampling_params = SamplingParams(
                n=n,
                temperature=temperature,
                top_p=top_p,
                use_beam_search=use_beam_search,
                stop=list(stop),
                stop_token_ids=stop_token_ids,
                max_tokens=max_new_tokens,
                top_k=top_k,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                best_of=best_of,
                logprobs=0,
                prompt_logprobs=0,
                include_stop_str_in_output=include_stop_str_in_output,
                seed=seed,
            )
            results_generator = engine.generate(context, sampling_params, request_id)
            async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
                logger.info(f"Updated outputs: {request_output}")
                prompt = request_output.prompt
                if echo:
                    text_outputs = [
                        prompt + output.text for output in request_output.outputs
                    ]
                else:
                    text_outputs = [output.text for output in request_output.outputs]
                # text_outputs = " ".join(text_outputs)
                # Note: usage is not supported yet
                prompt_tokens = len(request_output.prompt_token_ids)
                completion_tokens = sum(
                    len(output.token_ids) for output in request_output.outputs
                )
                # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
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
                    # "prompt_token": prompt_logprob[0],
                    "prompt_logprob":  extract_first_logprob(request_output.prompt_logprobs),
                    "output_token_len": [
                        len(output.token_ids) for output in request_output.outputs
                    ],
                    "finish_reason": (
                        request_output.outputs[0].finish_reason
                        if len(request_output.outputs) == 1
                        else [output.finish_reason for output in request_output.outputs]
                    ),
                }
                # logger.info(f"Updated ret: {ret}")
                yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔

        else:
            sampling_params = SamplingParams(
                n=n,
                temperature=temperature,
                top_p=top_p,
                use_beam_search=use_beam_search,
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
            async for request_output in results_generator: # 使用异步迭代器 results_generator 逐个获取生成的请求输出
                logger.info(f"Updated outputs: {request_output}")
                prompt = request_output.prompt
                if echo:
                    text_outputs = [
                        prompt + output.text for output in request_output.outputs
                    ]
                else:
                    text_outputs = [output.text for output in request_output.outputs]
                # text_outputs = " ".join(text_outputs)
                # Note: usage is not supported yet
                prompt_tokens = len(request_output.prompt_token_ids)
                completion_tokens = sum(
                    len(output.token_ids) for output in request_output.outputs
                )
                # logger.info(f"Updated prompt_logprob: {prompt_logprob}")
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
                # logger.info(f"Updated ret: {ret}")
                yield (json.dumps(ret) + "\0").encode() # 将字典转换为 JSON 格式，并以字节流的形式返回，每个响应以 \0 字符分隔
                
    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        # logger.info(f"Updated out: {x[:-1]}")
        return json.loads(x[:-1].decode()) # 将去掉最后一个字符后的字节字符串解码为普通字符串


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

# 模型调用时会对应到这个接口
@app.post("/worker_generate")
async def api_generate(request: Request):
    # 从请求中读取 JSON 数据并解析为一个字典 params
    params = await request.json()
    # 用于获取信号量，限制同时处理的请求数量，防止资源过载
    await acquire_worker_semaphore()
    # 生成一个唯一的请求ID，并将其添加到 params 字典中
    request_id = random_uuid()
    params["request_id"] = request_id
    output = await worker.generate(params)
    # 释放信号量，允许其他请求进入
    release_worker_semaphore()
    # 调用 engine 对象的 abort 方法，传入请求ID，用于终止与该请求相关的任何未完成任务
    await engine.abort(request_id)
    # 将生成的结果 output 封装成 JSON 响应并返回给客户端
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
    # print(args.speculative_model_path)
    if args.speculative_model_path:
        args.speculative_model = args.speculative_model_path
        args.num_speculative_tokens = 11
        args.use_v2_block_manager = True
    if args.num_gpus > 1:
        args.tensor_parallel_size = args.num_gpus

    engine_args = AsyncEngineArgs.from_cli_args(args)
    # logger.info(f"Updated engine_args: {engine_args}")
    # engine_args.max_model_len = 8192
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
