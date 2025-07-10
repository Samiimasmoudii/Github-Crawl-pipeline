import abc
import numpy as np
import copy
import pdb
import torch
import time
import os
from vllm import LLM, SamplingParams
import functools
from distributed.utils import print_with_rank
from transformers import PreTrainedTokenizer
from reason.inference.lm_call import LMCallingConfig, ConcatedLMGenResult, LanguageModelCallingFunction, VLLMRemoteCaller, VLLMSpeculativeRemoteCaller
import logging
import json
import re
import gc
from sklearn.metrics import mean_squared_error, mean_absolute_error
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RewardModelBaseConfig,
    RemoteRewardModelConfig,
)
from fastchat.model.model_adapter import (
    load_model,
    add_model_args,
    get_generate_stream_function,
)
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig
from typing import List, Optional, Tuple, Union, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import warnings
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime
import math

# Get the current date and time and format it as a string
current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# Add date information to log file name
log_filename = f'log/openr_base_env_log_{current_datetime}.log'
# Configuring logging
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



INVALID_ANS = "[invalid]"

import random
random.seed(42)

def accept_value_ablation(x, p):
    """
    Decide whether to accept value x given a threshold c and probability p.

    Parameters:
    x (float): The value to evaluate.
    c (float): The threshold.
    p (float): The probability of accepting x when x is greater than c; when x is less than or equal to c, the probability of accepting x is 1-p.

    Returns:
    bool: True for acceptance, False for rejection.
    """
    return random.random() < p

def filter_and_select_top_n_ablation(lst, p, n):
    """
    Given an input list containing sublists, filter the last element of the sublist according to the given threshold c and probability p, and select the original sublist indices corresponding to the first N values.

    Parameters:
    lst (list of lists of float): list containing sublists, each with at least one element.
    c (float): threshold.
    p (float): probability of accepting a value when it is greater than c.
    n (int): number of values ​​to be selected in the end.

    Returns:
    list of int: list containing the original sublist indices corresponding to the first N values selected.
    """
    # Create a list containing (last element of sublist, sublist index) and filter
    indexed_last_elements = [(sublist[-1], idx) for idx, sublist in enumerate(lst) 
                             if isinstance(sublist, list) and sublist and accept_value_ablation(sublist[-1], p)]
    
    # If the number of accepted values ​​exceeds N, the first N values ​​are selected after sorting by value
    if len(indexed_last_elements) > n:
        indexed_last_elements.sort(key=lambda x: x[0], reverse=True)  # Sort by value from largest to smallest
        indexed_last_elements = indexed_last_elements[:n]
    
    # Extract and return the index
    selected_indices = [idx for _, idx in indexed_last_elements]
    
    return selected_indices

def find_indices(lst, threshold):
    if not lst or len(lst) == 0:
        return []

    # Create a list containing (last element of sublist, sublist index)
    indexed_last_elements = [(sublist[-1], idx) for idx, sublist in enumerate(lst) if
                             isinstance(sublist, list) and sublist]

    # Checks how many sublists have their last element greater than a given threshold
    above_threshold = [idx for value, idx in indexed_last_elements if value >= threshold]
    # print(len(indexed_last_elements)/2)
    if len(above_threshold) >= len(indexed_last_elements)/2:
        sorted_above_threshold = sorted(above_threshold, key=lambda i: lst[i][-1], reverse=True)
        result_indices = sorted(sorted_above_threshold[:int(len(indexed_last_elements)/2)])
    else:
        # Otherwise, directly return all sublist indices greater than the threshold
        result_indices = above_threshold

    return result_indices


class NoLegalActionException(Exception):
    pass


class ResetException(Exception):
    pass


class BaseEnv(abc.ABC):
    """Basic environment to use for MCTS"""

    @abc.abstractmethod
    def reset(self, update_legal_action: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def legal_actions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    @staticmethod
    def build_query_str(
        cot_task_desc: Optional[str],
        cot_examples: Optional[str],
        problem_format_str: str,
        problem_input: str,
        is_few_shot: bool = False,
    ):
        """a wrap function that wrap the problem text with certrain format
        e.g. prompt_str = "Input: " + join_numbers(" ", xs) + "\nSteps:\n"
        >>> query_str = Game24Env.build_query_str("1 1 1 1")
        >>> print(query_str)
        >>> Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
        Input: 1 1 1 1
        Steps:

        >>>
        """

        ret = ""
        if cot_task_desc:
            ret += cot_task_desc + "\n"
        if is_few_shot:
            ret += cot_examples + "\n"
        ret += problem_format_str.format(question=problem_input)

        return ret

    @staticmethod
    def build_response_str(
        answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool
    ):
        raise NotImplementedError


class CoTEnv(BaseEnv):
    """The basic environment for solving natural language problems using CoT"""

    sep: str

    @property
    def stop_str(self):
        return NotImplementedError

    def _is_correct(self, completion) -> bool:
        raise NotImplementedError

    def get_reward(self):
        """To implement based on learned reward model"""
        raise NotImplementedError

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn: LanguageModelCallingFunction,
        appro_gen_fn: LanguageModelCallingFunction,
        speculative_gen_fn: LanguageModelCallingFunction,
        rm_gen_fn,
        task_desc_str: str,
        cot_example_str: str,
        problem_format_str: str,
        reset=True,
        seed=None,
        c=None,
        task_name=None,
        baseline_mode=None,
        model_mode=None,
        N=None,
    ):
        self.config = config
        self.mcts_mode = "play_with_bot_mode"
        self.math_problems = math_problems
        self.device = 'cuda'
        self.seed = seed
        self.c = c
        self.N = N
        self.task = task_name
        self.baseline_mode = baseline_mode
        self.model_mode = model_mode
        self.llm_gen_fn = llm_gen_fn
        self.appro_gen_fn = appro_gen_fn
        self.speculative_gen_fn = speculative_gen_fn
        self.rm_gen_fn = rm_gen_fn

        self.action_history = None
        self.reward_history = None
        self.large_reward = None
        self.used_fallback = None
        self.dynamic_threshold = None
        self.task_name = 'MATH'
        self.estimate_threshold = None
        self.math_problem = None
        self._legal_actions = None
        self.prm = None
        self.is_few_shot = config.get("is_few_shot", False)

        self._task_desc_str = task_desc_str
        self._cot_example_str = cot_example_str
        self._problem_format_str = problem_format_str

        prefixes = []
        if self._task_desc_str is not None:
            prefixes.append(self._task_desc_str)
        if self.is_few_shot:
            prefixes.append(self._cot_example_str)
        if len(prefixes) > 0:
            self.task_prefix = "\n".join(prefixes)
        else:
            self.task_prefix = None
        if reset:
            self.reset(True, N=N, seed=seed, c=c, baseline_mode=baseline_mode, full_reward=1, x=0.9)
    
    def reset(self, update_legal_action=True, N=6, seed=None, c=0, baseline_mode=None, full_reward=None, x=None):
        # reset environment to problem idx
        self.set_problem(idx=0)
        self.action_history = []
        self.reward_history = []
        self.large_reward = []
        self.mean = 0
        self.used_fallback = False
        self._init_query = self.build_query_str(
            cot_examples=self._cot_example_str,
            cot_task_desc=self._task_desc_str,
            problem_format_str=self._problem_format_str,
            problem_input=self.math_problem["question"],
            is_few_shot=self.is_few_shot,
        )
        if update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions,  api_completion_token, total_time, reward_time, verifiction_time = (
                        self.update_legal_actions(N, seed, c, baseline_mode, full_reward, x)
                    )
                    break
                except NoLegalActionException as e:
                    if cnt == 3:
                         raise ResetException
        info = {"api_completion_token": api_completion_token}
        torch.cuda.empty_cache()
        return self.get_state(), info, total_time, reward_time, verifiction_time

    def step(self, action, reward, reward_average, update_legal_action=True, N=6,  seed=None, c=0, task_name=None, baseline_mode=None, model_mode=None, full_reward=None, x=None):
        total_time = 0
        reward_time = 0
        verification_time = 0
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.large_reward.append(reward_average)
        state = self.get_state()
        reward = self.get_reward()
        terminated, truncated, info = self.get_done_and_info(task_name, model_mode)
        if len("".join(self.action_history)) >= 4000:
            terminated = True
        # update legal actions
        if not (terminated or truncated) and update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token, total_time, reward_time, verification_time = self.update_legal_actions(N, seed, c, baseline_mode, full_reward, x)
                    info["api_completion_token"] = api_completion_token
                    break # supplementary code
                except NoLegalActionException as e:
                    if cnt == 3:
                        terminated = True
                    info["api_completion_token"] = api_completion_token
                    break # supplementary code
                except NoLegalActionException as e:
                    if cnt == 3:
                        terminated = True
                        reward = 0
                        self._legal_actions = None
                        info["winner"] = 2
                        info["api_completion_token"] = 0
                    else:
                        pass
        else:
            self._legal_actions = None
            if info["winner"] == 1:
                reward = 1.0
            info["api_completion_token"] = 0
        torch.cuda.empty_cache()
        return state, reward, terminated, truncated, info, total_time, reward_time, verification_time

    def get_state(self):
        # not join about sep_str here because we let vllm return with sep_str
        ret = self._init_query + "".join(self.action_history)
        return ret

    def post_process_act(self, action: str):
        # This step may change the token count
        return action
    
    def remove_extra_newlines(self, action: str):

        parts = action.split('\n\n', 1)

        if len(parts) > 1:
            result = parts[0] + '\n\n'
        else:
            result = action

        return result
    
    def update_legal_actions(self, N, seed, c, baseline_mode, full_reward, x):
        total_time = 0
        reward_time = 0
        verification_time = 0
        score = 0
        texts = []
        logps_avg_by_len = []
        token_len = []
        completion_tokens = 0
        finish_reason = []
        reward_list = []
        result_s_text = []
        result_s_finish_reason = []
        result_l_text = []
        result_l_finish_reason = []
        result_l_num_tokens = []
        result_s_num_tokens = []
        result_s_prompt_logprob = []
        large_reward = 0.0
        large_list = []
        draft_time = 0
        speculative_time = 0

        # baseline_mode : 1-5 for baselines, 6 for initial method using a fixed threshold, 7 for our SpecSearch, 8-10 for ablation study

        if baseline_mode == 1:
            # large model parallel
            result_l: ConcatedLMGenResult = self.llm_gen_fn(
                input_str=self.get_state(),
                seed=seed,
                config=LMCallingConfig(
                    n=N,
                    stop_str=self.sep,
                    include_stop_str_in_output=True,
                    **self.config["generation_config"]
                ),
            )
            total_time += result_l.total_time
            start_time = time.time()
            prm_l = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                [
                    (
                        self.question,
                        self.answer + x,
                    )
                    for x in result_l.text
                ]
            )
            reward_time = time.time() - start_time
            for i in range(N):
                reward_list.append(prm_l[i][-1])
                texts.append(result_l.text[i])
                logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                token_len.append(result_l.num_tokens[i])
                finish_reason.append(result_l.finish_reason[i])
            completion_tokens = sum(token_len) 
        elif baseline_mode == 2:
            # large model serial
            for i in range(N):
                result_l: ConcatedLMGenResult = self.llm_gen_fn(
                    input_str=self.get_state(),
                    seed=seed+i,
                    config=LMCallingConfig(
                        n=1,
                        stop_str=self.sep,
                        include_stop_str_in_output=True,
                        **self.config["generation_config"]
                    ),
                )
                result_l_text.append(result_l.text[0])
                result_l_num_tokens.append(result_l.num_tokens[0])
                result_l_finish_reason.append(result_l.finish_reason)
                total_time += result_l.total_time
            start_time = time.time()
            prm_l = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                [
                    (
                        self.question,
                        self.answer + x,
                    )
                    for x in result_l_text
                ]
            )
            reward_time = time.time() - start_time
            for i in range(N):
                reward_list.append(prm_l[i][-1])
                texts.append(result_l_text[i])
                logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                token_len.append(result_l_num_tokens[i])
                finish_reason.append(result_l_finish_reason[i])
            completion_tokens = sum(token_len)
        elif baseline_mode == 3:
            # draft model parallel
            result_s: ConcatedLMGenResult = self.appro_gen_fn(
                input_str=self.get_state(),
                seed=seed,
                config=LMCallingConfig(
                    n=N,
                    stop_str=self.sep,
                    include_stop_str_in_output=True,
                    **self.config["generation_config"]
                ),
            )
            total_time += result_s.total_time
            start_time = time.time()
            prm_s = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                [
                    (
                        self.question,
                        self.answer + "".join(x),
                    )
                    for x in result_s.text
                ]
            )
            reward_time = time.time() - start_time
            for i in range(N):
                reward_list.append(prm_s[i][-1])
                texts.append("".join(result_s.text[i]))
                logps_avg_by_len.append(result_s.logp_avg_by_len[0])
                token_len.append(result_s.num_tokens[i])
                finish_reason.append(result_s.finish_reason[i])
            completion_tokens = sum(token_len)
            print(self.get_state())
        elif baseline_mode == 4:
            # draft model serial
            for i in range(N):
                result_s: ConcatedLMGenResult = self.appro_gen_fn(
                    input_str=self.get_state(),
                    seed=seed+i,
                    config=LMCallingConfig(
                        n=1,
                        stop_str=self.sep,
                        include_stop_str_in_output=True,
                        **self.config["generation_config"]
                    ),
                )
                result_s_text.append(result_s.text)
                result_s_num_tokens.append(result_s.num_tokens[0])
                result_s_prompt_logprob.append(result_s.prompt_logprob)
                result_s_finish_reason.append(result_s.finish_reason)
                total_time += result_s.total_time
            start_time = time.time()
            prm_s = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                [
                    (
                        self.question,
                        self.answer + "".join(x),
                    )
                    for x in result_s_text
                ]
            )
            reward_time = time.time() - start_time
            for i in range(N):
                reward_list.append(prm_s[i][-1])
                texts.append("".join(result_s_text[i]))
                logps_avg_by_len.append(result_s.logp_avg_by_len[0])
                token_len.append(result_s_num_tokens[i])
                finish_reason.append(result_s_finish_reason[i])
            completion_tokens = sum(token_len) 
        elif baseline_mode == 5:
            # speculative model serial
            for i in range(N):
                result_l: ConcatedLMGenResult = self.speculative_gen_fn(
                    input_str=self.get_state(),
                    seed=seed+i,
                    config=LMCallingConfig(
                        n=1,
                        stop_str=self.sep,
                        include_stop_str_in_output=True,
                        **self.config["generation_config"]
                    ),
                )
                result_l_text.append(result_l.text[0])
                result_l_num_tokens.append(result_l.num_tokens[0])
                result_l_finish_reason.append(result_l.finish_reason)
                total_time += result_l.total_time
            start_time = time.time()
            prm_l = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                [
                    (
                        self.question,
                        self.answer + x,
                    )
                    for x in result_l_text
                ]
            )
            reward_time = time.time() - start_time
            for i in range(N):
                reward_list.append(prm_l[i][-1])
                texts.append(result_l_text[i])
                logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                token_len.append(result_l_num_tokens[i])
                finish_reason.append(result_l_finish_reason[i])
            completion_tokens = sum(token_len)
        elif baseline_mode == 6:
            # Fixed Threshold
            logging.info(f"Updated threshold: {c}")
            action_list = []
            result_s: ConcatedLMGenResult = self.appro_gen_fn(
                input_str=self.get_state(),
                seed=seed,
                config=LMCallingConfig(
                    n=2*N,
                    stop_str=self.sep,
                    include_stop_str_in_output=True,
                    **self.config["generation_config"]
                ),
            )
            total_time += result_s.total_time
            draft_time += result_s.total_time
            start_time = time.time()
            prm_s = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                [
                    (
                        self.question,
                        self.answer + "".join(x),
                    )
                    for x in result_s.text
                ]
            )
            reward_time += time.time() - start_time
            indices = find_indices(prm_s, c)
            for i in indices:
                reward_list.append(prm_s[i][-1])
                texts.append("".join(result_s.text[i]))
                logps_avg_by_len.append(result_s.logp_avg_by_len[0])
                token_len.append(result_s.num_tokens[i])
                finish_reason.append(result_s.finish_reason[i])
            score = len(indices)
            if score == 0:
                reward_list = []
                texts = []
                logps_avg_by_len = []
                token_len = []
                finish_reason = []
                for i in range(N):
                    result_l: ConcatedLMGenResult = self.speculative_gen_fn(
                        input_str=self.get_state(),
                        seed=seed+i,
                        config=LMCallingConfig(
                            n=1,
                            stop_str=self.sep,
                            include_stop_str_in_output=True,
                            **self.config["generation_config"]
                        ),
                    )
                    result_l_text.append(result_l.text[0])
                    result_l_num_tokens.append(result_l.num_tokens[0])
                    result_l_finish_reason.append(result_l.finish_reason)
                    total_time += result_l.total_time
                    speculative_time += result_l.total_time
                start_time = time.time()
                prm_l = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                    [
                        (
                            self.question,
                            self.answer + x,
                        )
                        for x in result_l_text
                    ]
                )
                reward_time = time.time() - start_time
                for i in range(N):
                    reward_list.append(prm_l[i][-1])
                    texts.append(result_l_text[i])
                    logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                    token_len.append(result_l_num_tokens[i])
                    finish_reason.append(result_l_finish_reason[i])
            elif score < N:
                for i in range(N-score):
                    result_l: ConcatedLMGenResult = self.speculative_gen_fn(
                        input_str=self.get_state(),
                        seed=seed+i,
                        config=LMCallingConfig(
                            n=1,
                            stop_str=self.sep,
                            include_stop_str_in_output=True,
                            **self.config["generation_config"]
                        ),
                    )
                    result_l_text.append(result_l.text[0])
                    result_l_num_tokens.append(result_l.num_tokens[0])
                    result_l_finish_reason.append(result_l.finish_reason)
                    total_time += result_l.total_time
                    speculative_time += result_l.total_time
                    action_list.append(result_l.text[0])
                    texts.append(result_l.text[0])
                    logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                    token_len.append(result_l.num_tokens[0])
                    finish_reason.append(result_l.finish_reason[0])
                start_time = time.time()
                prm_m = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                    [
                        (
                            self.question,
                            self.answer + x,
                        )
                        for x in action_list
                    ]
                )
                reward_time += time.time() - start_time
                for i in range(N-len(indices)):
                    reward_list.append(prm_m[i][-1])
            completion_tokens = sum(token_len)
        elif baseline_mode == 7:
            # SpecSearch: Our approach
            r = 1.0
            # Calculate the current threshold
            if len(self.reward_history) >= 1:
                # initialize a0
                a_prev = self.large_reward[0] * 0.9

                if len(self.reward_history) >= 2:
                    for i in range(1, len(self.large_reward)):
                        if self.large_reward[i] != 0:
                            a_next = (a_prev * x + (1 - x) * self.large_reward[i]) * r
                        else:
                            a_next = a_prev
                        a_prev = a_next
                c = a_prev
            logging.info(f"Updated threshold: {c}")
            # The root nodes are uniformly generated using the speculative model
            if len(self.reward_history) == 0:
                # speculative model serial
                for i in range(N):
                    result_l: ConcatedLMGenResult = self.speculative_gen_fn(
                        input_str=self.get_state(),
                        seed=seed+i,
                        config=LMCallingConfig(
                            n=1,
                            stop_str=self.sep,
                            include_stop_str_in_output=True,
                            **self.config["generation_config"]
                        ),
                    )
                    result_l_text.append(result_l.text[0])
                    result_l_num_tokens.append(result_l.num_tokens[0])
                    result_l_finish_reason.append(result_l.finish_reason)
                    total_time += result_l.total_time
                start_time = time.time()
                prm_l = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                    [
                        (
                            self.question,
                            self.answer + x,
                        )
                        for x in result_l_text
                    ]
                )
                reward_time = time.time() - start_time
                for i in range(N):
                    reward_list.append(prm_l[i][-1])
                    large_list.append(prm_l[i][-1])
                    texts.append(result_l_text[i])
                    logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                    token_len.append(result_l_num_tokens[i])
                    finish_reason.append(result_l_finish_reason[i])
                completion_tokens = sum(token_len)
                self.estimate_threshold = sum(reward_list)/len(reward_list)
                large_reward = max(reward_list)
            else:
                action_list = []
                result_s: ConcatedLMGenResult = self.appro_gen_fn(
                    input_str=self.get_state(),
                    seed=seed,
                    config=LMCallingConfig(
                        n=2*N,
                        stop_str=self.sep,
                        include_stop_str_in_output=True,
                        **self.config["generation_config"]
                    ),
                )
                total_time += result_s.total_time
                draft_time += result_s.total_time
                start_time = time.time()
                prm_s = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                    [
                        (
                            self.question,
                            self.answer + "".join(x),
                        )
                        for x in result_s.text
                    ]
                )
                reward_time += time.time() - start_time
                # Determine the thoughts to be retained based on the threshold
                indices = find_indices(prm_s, c)
                for i in indices:
                    reward_list.append(prm_s[i][-1])
                    texts.append("".join(result_s.text[i]))
                    logps_avg_by_len.append(result_s.logp_avg_by_len[0])
                    token_len.append(result_s.num_tokens[i])
                    finish_reason.append(result_s.finish_reason[i])
                score = len(indices)
                if score == 0:
                    reward_list = []
                    texts = []
                    logps_avg_by_len = []
                    token_len = []
                    finish_reason = []
                    for i in range(N):
                        result_l: ConcatedLMGenResult = self.speculative_gen_fn(
                            input_str=self.get_state(),
                            seed=seed+i,
                            config=LMCallingConfig(
                                n=1,
                                stop_str=self.sep,
                                include_stop_str_in_output=True,
                                **self.config["generation_config"]
                            ),
                        )
                        result_l_text.append(result_l.text[0])
                        result_l_num_tokens.append(result_l.num_tokens[0])
                        result_l_finish_reason.append(result_l.finish_reason)
                        total_time += result_l.total_time
                        speculative_time += result_l.total_time
                    start_time = time.time()
                    prm_l = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                        [
                            (
                                self.question,
                                self.answer + x,
                            )
                            for x in result_l_text
                        ]
                    )
                    reward_time = time.time() - start_time
                    for i in range(N):
                        reward_list.append(prm_l[i][-1])
                        large_list.append(prm_l[i][-1])
                        texts.append(result_l_text[i])
                        logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                        token_len.append(result_l_num_tokens[i])
                        finish_reason.append(result_l_finish_reason[i])
                elif score < N:
                    for i in range(N-score):
                        result_l: ConcatedLMGenResult = self.speculative_gen_fn(
                            input_str=self.get_state(),
                            seed=seed+i,
                            config=LMCallingConfig(
                                n=1,
                                stop_str=self.sep,
                                include_stop_str_in_output=True,
                                **self.config["generation_config"]
                            ),
                        )
                        result_l_text.append(result_l.text[0])
                        result_l_num_tokens.append(result_l.num_tokens[0])
                        result_l_finish_reason.append(result_l.finish_reason)
                        total_time += result_l.total_time
                        speculative_time += result_l.total_time
                        action_list.append(result_l.text[0])
                        texts.append(result_l.text[0])
                        logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                        token_len.append(result_l.num_tokens[0])
                        finish_reason.append(result_l.finish_reason[0])
                    start_time = time.time()
                    prm_m = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                        [
                            (
                                self.question,
                                self.answer + x,
                            )
                            for x in action_list
                        ]
                    )
                    reward_time += time.time() - start_time
                    for i in range(N-len(indices)):
                        reward_list.append(prm_m[i][-1])
                        large_list.append(prm_m[i][-1])
                completion_tokens = sum(token_len)
                if full_reward:
                    large_reward = max(reward_list)
                else:
                    if len(large_list) > 0:
                        large_reward = max(large_list)
        elif baseline_mode == 8:
            # The large model generates N/2 thoughts, and the small model generates N/2 thoughts
            score = 3
            action_list = []
            result_s: ConcatedLMGenResult = self.appro_gen_fn(
                input_str=self.get_state(),
                seed=seed,
                config=LMCallingConfig(
                    n=3,
                    stop_str=self.sep,
                    include_stop_str_in_output=True,
                    **self.config["generation_config"]
                ),
            )
            total_time += result_s.total_time
            draft_time += result_s.total_time
            for i in range(3):
                texts.append("".join(result_s.text[i]))
                logps_avg_by_len.append(result_s.logp_avg_by_len[0])
                token_len.append(result_s.num_tokens[i])
                finish_reason.append(result_s.finish_reason[i])
            for i in range(3):
                result_l: ConcatedLMGenResult = self.speculative_gen_fn(
                    input_str=self.get_state(),
                    seed=seed+i,
                    config=LMCallingConfig(
                        n=1,
                        stop_str=self.sep,
                        include_stop_str_in_output=True,
                        **self.config["generation_config"]
                    ),
                )
                result_l_text.append(result_l.text[0])
                result_l_num_tokens.append(result_l.num_tokens[0])
                result_l_finish_reason.append(result_l.finish_reason)
                total_time += result_l.total_time
            for i in range(3):
                texts.append(result_l_text[i])
                logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                token_len.append(result_l_num_tokens[i])
                finish_reason.append(result_l_finish_reason[i])
            start_time = time.time()
            prm_l = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                [
                    (
                        self.question,
                        self.answer + x,
                    )
                    for x in texts
                ]
            )
            reward_time = time.time() - start_time
            for i in range(N):
                reward_list.append(prm_l[i][-1])
            completion_tokens = sum(token_len)

        elif baseline_mode == 9:
            # p = 50% accept small model thought N randomly
            logging.info(f"Updated threshold: {c}")
            action_list = []
            result_s: ConcatedLMGenResult = self.appro_gen_fn(
                input_str=self.get_state(),
                seed=seed,
                config=LMCallingConfig(
                    n=N,
                    stop_str=self.sep,
                    include_stop_str_in_output=True,
                    **self.config["generation_config"]
                ),
            )
            total_time += result_s.total_time
            draft_time += result_s.total_time
            start_time = time.time()
            prm_s = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                [
                    (
                        self.question,
                        self.answer + "".join(x),
                    )
                    for x in result_s.text
                ]
            )
            reward_time += time.time() - start_time
            random.seed(42)
            indices = filter_and_select_top_n_ablation(prm_s, 0.5, N)
            for i in indices:
                reward_list.append(prm_s[i][-1])
                texts.append("".join(result_s.text[i]))
                logps_avg_by_len.append(result_s.logp_avg_by_len[0])
                token_len.append(result_s.num_tokens[i])
                finish_reason.append(result_s.finish_reason[i])
            score = len(indices)
            if score == 0:
                reward_list = []
                texts = []
                logps_avg_by_len = []
                token_len = []
                finish_reason = []
                for i in range(N):
                    result_l: ConcatedLMGenResult = self.speculative_gen_fn(
                        input_str=self.get_state(),
                        seed=seed+i,
                        config=LMCallingConfig(
                            n=1,
                            stop_str=self.sep,
                            include_stop_str_in_output=True,
                            **self.config["generation_config"]
                        ),
                    )
                    result_l_text.append(result_l.text[0])
                    result_l_num_tokens.append(result_l.num_tokens[0])
                    result_l_finish_reason.append(result_l.finish_reason)
                    total_time += result_l.total_time
                    speculative_time += result_l.total_time
                start_time = time.time()
                prm_l = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                    [
                        (
                            self.question,
                            self.answer + x,
                        )
                        for x in result_l_text
                    ]
                )
                reward_time = time.time() - start_time
                for i in range(N):
                    reward_list.append(prm_l[i][-1])
                    large_list.append(prm_l[i][-1])
                    texts.append(result_l_text[i])
                    logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                    token_len.append(result_l_num_tokens[i])
                    finish_reason.append(result_l_finish_reason[i])
            elif score < N:
                for i in range(N-score):
                    result_l: ConcatedLMGenResult = self.speculative_gen_fn(
                        input_str=self.get_state(),
                        seed=seed+i,
                        config=LMCallingConfig(
                            n=1,
                            stop_str=self.sep,
                            include_stop_str_in_output=True,
                            **self.config["generation_config"]
                        ),
                    )
                    result_l_text.append(result_l.text[0])
                    result_l_num_tokens.append(result_l.num_tokens[0])
                    result_l_finish_reason.append(result_l.finish_reason)
                    total_time += result_l.total_time
                    speculative_time += result_l.total_time
                    action_list.append(result_l.text[0])
                    texts.append(result_l.text[0])
                    logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                    token_len.append(result_l.num_tokens[0])
                    finish_reason.append(result_l.finish_reason[0])
                start_time = time.time()
                prm_m = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                    [
                        (
                            self.question,
                            self.answer + x,
                        )
                        for x in action_list
                    ]
                )
                reward_time += time.time() - start_time
                for i in range(N-len(indices)):
                    reward_list.append(prm_m[i][-1])
                    large_list.append(prm_m[i][-1])
            completion_tokens = sum(token_len)
            if full_reward:
                large_reward = max(reward_list)
            else:
                if len(large_list) > 0:
                    large_reward = max(large_list)
        elif baseline_mode == 10:
            # logprob 
            action_list = []
            sum_logprob = []
            result_s: ConcatedLMGenResult = self.appro_gen_fn(
                input_str=self.get_state(),
                seed=seed,
                config=LMCallingConfig(
                    n=2*N,
                    stop_str=self.sep,
                    include_stop_str_in_output=True,
                    **self.config["generation_config"]
                ),
            )
            total_time += result_s.total_time
            draft_time += result_s.total_time
            start_time = time.time()
            prm_s = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                [
                    (
                        self.question,
                        self.answer + "".join(x),
                    )
                    for x in result_s.text
                ]
            )
            reward_time += time.time() - start_time
            indices = []
            for i in range(2*N):
                result_score: ConcatedLMGenResult = self.llm_gen_fn(
                    input_str=self.get_state()+"".join(result_s.text[i]),
                    seed=seed,
                    config=LMCallingConfig(
                        n=1,
                        stop_str=self.sep,
                        include_stop_str_in_output=True,
                        max_new_tokens=1,
                        temperature=0.7,
                        top_p=1.0,
                        top_k=-1,
                    ),
                )
                total_time += result_score.total_time
                prompt_logprob = result_score.prompt_logprob
                sum_logprob.append(sum(prompt_logprob[-result_s.num_tokens[i]:]))
                if sum_logprob[i] < result_s.cumulative_logprob[i]:
                    indices.append(i)
            if len(indices) > N:
                top_n_indices = sorted(indices, key=lambda i: prm_s[i][-1], reverse=True)[:N]
                for i in top_n_indices:
                    reward_list.append(prm_s[i][-1])
                    texts.append("".join(result_s.text[i]))
                    logps_avg_by_len.append(result_s.logp_avg_by_len[0])
                    token_len.append(result_s.num_tokens[i])
                    finish_reason.append(result_s.finish_reason[i])
            else:
                for i in indices:
                    reward_list.append(prm_s[i][-1])
                    texts.append("".join(result_s.text[i]))
                    logps_avg_by_len.append(result_s.logp_avg_by_len[0])
                    token_len.append(result_s.num_tokens[i])
                    finish_reason.append(result_s.finish_reason[i])

            score = len(indices)
            if score > N:
                score = N
            if score == 0:
                reward_list = []
                texts = []
                logps_avg_by_len = []
                token_len = []
                finish_reason = []
                for i in range(N):
                    result_l: ConcatedLMGenResult = self.llm_gen_fn(
                        input_str=self.get_state(),
                        seed=seed+i,
                        config=LMCallingConfig(
                            n=1,
                            stop_str=self.sep,
                            include_stop_str_in_output=True,
                            **self.config["generation_config"]
                        ),
                    )
                    result_l_text.append(result_l.text[0])
                    result_l_num_tokens.append(result_l.num_tokens[0])
                    result_l_finish_reason.append(result_l.finish_reason)
                    total_time += result_l.total_time
                    speculative_time += result_l.total_time
                start_time = time.time()
                prm_l = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                    [
                        (
                            self.question,
                            self.answer + x,
                        )
                        for x in result_l_text
                    ]
                )
                reward_time = time.time() - start_time
                for i in range(N):
                    reward_list.append(prm_l[i][-1])
                    large_list.append(prm_l[i][-1])
                    texts.append(result_l_text[i])
                    logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                    token_len.append(result_l_num_tokens[i])
                    finish_reason.append(result_l_finish_reason[i])
            elif score < N:
                for i in range(N-score):
                    result_l: ConcatedLMGenResult = self.llm_gen_fn(
                        input_str=self.get_state(),
                        seed=seed+i,
                        config=LMCallingConfig(
                            n=1,
                            stop_str=self.sep,
                            include_stop_str_in_output=True,
                            **self.config["generation_config"]
                        ),
                    )
                    result_l_text.append(result_l.text[0])
                    result_l_num_tokens.append(result_l.num_tokens[0])
                    result_l_finish_reason.append(result_l.finish_reason)
                    total_time += result_l.total_time
                    speculative_time += result_l.total_time
                    action_list.append(result_l.text[0])
                    texts.append(result_l.text[0])
                    logps_avg_by_len.append(result_l.logp_avg_by_len[0])
                    token_len.append(result_l.num_tokens[0])
                    finish_reason.append(result_l.finish_reason[0])
                start_time = time.time()
                prm_m = self.rm_gen_fn( # call reward_fn to calculate the reward for each legal action
                    [
                        (
                            self.question,
                            self.answer + x,
                        )
                        for x in action_list
                    ]
                )
                reward_time += time.time() - start_time
                for i in range(N-len(indices)):
                    reward_list.append(prm_m[i][-1])
                    large_list.append(prm_m[i][-1])
            completion_tokens = sum(token_len)
                # large_reward = sum(large_list)/len(large_list)
            if full_reward:
                large_reward = max(reward_list)
            else:
                if len(large_list) > 0:
                    large_reward = max(large_list)
        else:
            error_message = "Error: baseline_mode is wrong."
            raise ValueError(error_message)  # Throw an exception and terminate the program

        text_list, prob_list, num_token_list = [], [], []
        finish_reason_list = []
        next_state_terminated = {}

        # # parallel
        # completion_tokens = sum(result.num_tokens)
        for i in range(len(texts)):
            # XXX: this process can be improve or moved to other place
            # this is a pre-judge of terminal flag or certain action, by
            # whether the text-generation is stop by the <eos> or stop_str
            terminated = not texts[i].endswith(self.sep)
            processed_act = self.post_process_act(texts[i])
            if (
                len(processed_act) > 0
                and processed_act not in text_list
                # only stop is valid, otherwise the output action is truncated actually
                and finish_reason[i] == "stop" 
            ):
                text_list.append(processed_act)
                prob_list.append(logps_avg_by_len[i])
                num_token_list.append(token_len[i])
                finish_reason_list.append(finish_reason[i])
                next_state_terminated[processed_act] = terminated
   
        if len(prob_list) == 0:
            raise NoLegalActionException("No possible action have been generated.")

        prob_list = np.exp(prob_list)
        prob_list = np.array(prob_list)
        # normalize probability
        prob_list = prob_list / np.sum(prob_list)
        if full_reward:
            _legal_actions = [
                {
                    "action": action,
                    "prob": prob,
                    "num_token": n_token,
                    "finish_reason": finish_reason,
                    "reward": reward,
                    "large_reward": large_reward,
                    "reward_list": reward_list,
                    "threshold": c,
                }
                for action, prob, n_token, finish_reason, reward in zip(
                    text_list, prob_list, num_token_list, finish_reason_list, reward_list
                )
            ]
        else:
            _legal_actions = [
                {
                    "action": action,
                    "prob": prob,
                    "num_token": n_token,
                    "finish_reason": finish_reason,
                    "reward": reward,
                    "large_reward": large_reward,
                    "reward_list": large_list,
                    "threshold": c,
                }
                for action, prob, n_token, finish_reason, reward in zip(
                    text_list, prob_list, num_token_list, finish_reason_list, reward_list
                )
            ]
        self._next_state_terminated = next_state_terminated
        logging.info(f"Updated score: {score}")
        logging.info(f"Updated reward_list: {reward_list}")
        torch.cuda.empty_cache()
        return _legal_actions, completion_tokens, total_time, reward_time, verification_time

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    @property
    def query(self):
        return self._init_query
    
    @property
    def question(self)->str:
        return self.math_problem["question"]


    @property
    def answer(self):
        return "".join(self.action_history)
    
    def reward_list(self):
        return self.reward_history

    def get_done_and_info(self, task_name, model_mode):
        if model_mode == 2: # Llama model
            stop_str = "<|eot_id|>"
        else:
            stop_str = "<|im_end|>"
        info = {"winner": 0}
        # done when reaches maximum length or LLM generates stop words
        if stop_str is not None and stop_str in self.action_history[-1]:
            terminated = True
        elif self._next_state_terminated[self.action_history[-1]]:
            terminated = True
        elif self.sep not in self.action_history[-1]:
            # This is because the output is stopped by eos
            terminated = True
        else: terminated = False

        truncated = len(self.action_history) >= self.config["max_length"]
        assert len(self.action_history) <= self.config["max_length"]
        if terminated or truncated:
            if self._is_correct(self.action_history[-1]):
                info["winner"] = 1
            else:
                info["winner"] = 2
            return terminated, truncated, info
        return terminated, truncated, info

    def copy(self):
        env = self.__class__(
            self.config,
            self.math_problems,
            self.llm_gen_fn,
            self.appro_gen_fn ,
            self.speculative_gen_fn,
            self.rm_gen_fn,
            self._task_desc_str,
            self._cot_example_str,
            self._problem_format_str,
            reset=False,
            N=self.N,
            seed=self.seed,
            c=self.c,
            task_name=self.task,
            baseline_mode=self.baseline_mode,
            model_mode=self.model_mode,
        )
        env.math_problem = copy.deepcopy(self.math_problem)
        env._legal_actions = copy.deepcopy(self._legal_actions)
        env.action_history = copy.deepcopy(self.action_history)
        env.reward_history = copy.deepcopy(self.reward_history)
        env.large_reward = copy.deepcopy(self.large_reward)
        env.used_fallback = copy.deepcopy(self.used_fallback)
        env.dynamic_threshold = copy.deepcopy(self.dynamic_threshold)
        env.estimate_threshold = copy.deepcopy(self.estimate_threshold)
        env.mean = copy.deepcopy(self.mean)
        env._init_query = copy.deepcopy(self._init_query)
        env._next_state_terminated = copy.deepcopy(self._next_state_terminated)
        return env

    @property
    def legal_actions(self):
        return self._legal_actions