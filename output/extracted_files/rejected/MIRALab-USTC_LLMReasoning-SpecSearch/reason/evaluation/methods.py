from dataclasses import dataclass
import functools
from typing import Dict
from reason.inference.lm_call import LMCallingConfig, LanguageModelCallingFunction, ConcatedLMGenResult, BlockLMGenResult
from reason.inference.rm_call import RewardModelCallingFunction
from reason.evaluation.evaluator import SolutionOutput, Task, TreeSearchSolutionOutput
from reason.guided_search.tree import SearchTree
import torch
import numpy as np
import logging
import math
import time
logging.basicConfig(filename='openr_methods_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class BasicConfig:
    task_name: str

@dataclass
class CoTConfig(BasicConfig):
    pass

@dataclass
class TreeSearchConfig(BasicConfig):
    # construction config
    tree_max_width: int = 10
    tree_max_depth: int = 10
    # node config
    init_critic_value: bool = True

    def __post_init__(self):
        assert self.tree_max_width > 0, \
            "Tree width must be greater than 0"
        assert self.tree_max_depth > 0, \
            "Tree depth must be greater than 0"

@dataclass
class BeamSearchConfig(TreeSearchConfig):
    beam_size: int = 1 

    def __post_init__(self):
        super().__post_init__()
        assert self.beam_size > 0, \
            "Beam size must be greater than 0"
        assert self.init_critic_value, \
            "BeamSearch should set init_critic_value to True"
        
def beam_search(
    config: BeamSearchConfig,
    gen_config: LMCallingConfig,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    appro_call: LanguageModelCallingFunction,
    speculative_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
    seed,
    c,
    baseline_mode,
    model_mode,
    N,
    M,
    is_tensorboard,
    full_reward,
    idx,
    x,
) -> SolutionOutput:
    task = Task(task_name=config.task_name)
    flag = 0
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    if config.task_name in {'MATH100_qwen', 'MATH100_llama', 'GSM8K100_qwen', 'GSM8K100_llama', 'MATH50_qwen'}:
        env = task.env_fn(
            config={
                "max_actions": config.tree_max_width,
                "max_length": config.tree_max_depth,
                "stop_str": "The answer is ",
                "generation_config": {
                    "max_new_tokens": gen_config.max_new_tokens,
                    "temperature": gen_config.temperature,
                    "top_p": gen_config.top_p,
                    "top_k": gen_config.top_k,
                },
            },
            math_problems=[
                {
                    "question": problem_inst["question"],
                    "answer": task.extract_groundtruth(problem_inst["answer"]),
                }
            ],
            llm_gen_fn=lm_call,
            appro_gen_fn=appro_call,
            speculative_gen_fn=speculative_call,
            rm_gen_fn=rm_call_fn,
            seed=seed,
            c=c,
            task_name=flag,
            baseline_mode=baseline_mode,
            model_mode=model_mode,
            N=N,
            # TODO(ziyu): set sep by lm_call.lm_step_tag

        )
    else:
        error_message = "Error: dataset is wrong."
        raise ValueError(error_message)  # Throw an exception and terminate the program
    search_tree = SearchTree(cfg={})
    traj_list, total_time = search_tree.beam_search(
        env, config.beam_size, config.tree_max_depth, rm_call_fn, seed, c, config.task_name, baseline_mode, model_mode, N, M, is_tensorboard, full_reward, idx, x,
    )
    # logging.info(f"Updated test: {test}")
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        total_time=total_time,
    )

@dataclass
class MCTSBaseConfig(TreeSearchConfig):
    # PUCT hparams
    pb_c_base: float = 19652
    pb_c_init: float = 1.25

@dataclass
class VanilaMCTSConfig(MCTSBaseConfig):
    # rollout step strategy, if `select_by_prior` is False,
    #  then select by the initial critic value
    # otherwise, random choice by the prior probability
    select_by_prior: bool = False 
    num_path: int = 4
    
    def __post_init__(self):
        super().__post_init__()
        if not self.select_by_prior:
            assert self.init_critic_value, \
                "VanilaMCTS with greedy as rollout method should set init_critic_value to True"
        assert self.num_path > 0

def vanila_mcts(
    config: VanilaMCTSConfig,
    gen_config: LMCallingConfig,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    appro_call: LanguageModelCallingFunction,
    speculative_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
    seed,
    c,
    baseline_mode,
    model_mode,
    N,
    M,
    is_tensorboard,
    full_reward,
    idx,
    x,
):
    total_time = []
    task = Task(task_name=config.task_name)
    flag = 0
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    if config.task_name in {'MATH100_qwen', 'MATH100_llama', 'GSM8K100_qwen', 'GSM8K100_llama', 'MATH50_qwen'}:
        env = task.env_fn(
            config={
                "max_actions": config.tree_max_width,
                "max_length": config.tree_max_depth,
                "stop_str": "The answer is ",
                "generation_config": {
                    "max_new_tokens": gen_config.max_new_tokens,
                    "temperature": gen_config.temperature,
                    "top_p": gen_config.top_p,
                    "top_k": gen_config.top_k,
                },
            },
            math_problems=[
                {
                    "question": problem_inst["question"],
                    "answer": task.extract_groundtruth(problem_inst["answer"]),
                }
            ],
            llm_gen_fn=lm_call,
            appro_gen_fn=appro_call,
            speculative_gen_fn=speculative_call,
            rm_gen_fn=rm_call_fn,
            seed=seed,
            c=c,
            task_name=flag,
            baseline_mode=baseline_mode,
            model_mode=model_mode,
            N=N,
        )
    else:
        error_message = "Error: dataset is wrong."
        raise ValueError(error_message)  # Throw an exception and terminate the program

    search_tree = SearchTree(
        cfg={
            "pb_c_base": config.pb_c_base,
            "pb_c_init": config.pb_c_init,
            "init_critic_value": config.init_critic_value,
        }
    )
    traj_list,total_time = search_tree.vanila_mcts(
        simulate_env=env,
        num_path=config.num_path,
        reward_model_fn=rm_call_fn,
        seed=seed,
        c=c,
        task_name=config.task_name,
        baseline_mode=baseline_mode,
        model_mode=model_mode,
        N=N,
        M=M,
        is_tensorboard=is_tensorboard,
        full_reward=full_reward,
        idx=idx,
        x=x,
        select_by_prior=config.select_by_prior
    )
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        total_time=total_time,
    )
