from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from config.config_utils import str2bool
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller, VLLMSpeculativeRemoteCaller
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RewardModelBaseConfig,
    RemoteRewardModelConfig,
)
from reason.evaluation.evaluator import SolutionOutput, Task, RemoteMathEvaluator, MathEvaluator
import torch
from functools import partial
import json
import jsonlines
import time
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import os
import random
from multiprocessing import Pool
import tree
from ray.util.actor_pool import ActorPool
from reason.evaluation.methods import *
import ray
import logging

logging.basicConfig(filename='openr_evaluate_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')



def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

# Calculate the average
def calculate_average(lst):
    if lst:
        return sum(lst) / len(lst)
    return 0

if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument("--LM", type=str, required=True)
    parser.add_argument("--sLM", type=str, required=True)
    parser.add_argument("--RM", type=str, default="dummy")
    parser.add_argument("--controller_addr", type=str, default="http://0.0.0.0:28780")
    # task config
    parser.add_argument("--task_name", type=str, default="gsm8k")
    parser.add_argument("--test", type=str2bool, default=True)
    parser.add_argument("--is_few_shot", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--c", type=float, default=0.0)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--baseline_mode", type=int, default=None)
    parser.add_argument("--model_mode", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--is_tensorboard", type=int, default=None)
    parser.add_argument("--full_reward", type=int, default=None)
    # method config
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--num_sequence", type=int, default=1)
    # LM gen config
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    # Tree construction config
    parser.add_argument("--tree_max_depth", type=int, default=None)
    parser.add_argument("--tree_max_width", type=int, default=None)
    # ckpg config
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    # parallel config
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--num_worker", type=int, default=32)
    config = parser.parse_args()

    setup_seed(config.seed)

    ray.init(local_mode=True)
    config.num_worker = 1

    # TODO(ziyu): move into some configuration file
    if "math-shepherd" in config.RM.lower():
        prm_step_tag = "ки\n"
    else:
        # assume qwen
        prm_step_tag = "\n\n\n\n\n "
    prm_format_str = "{question} {answer}"

    if "qwen" in config.LM.lower():
        lm_step_tag = "\n\n"
    elif "llama" in config.LM.lower():
        lm_step_tag = "\n"
    else:
        lm_step_tag = "ки\n"

    # Initialize a remote caller for calling LLM
    llm_gen_fn = VLLMRemoteCaller(
        config.LM, config.controller_addr, lm_step_tag=lm_step_tag
    )
    appro_gen_fn = VLLMRemoteCaller(
        config.sLM, config.controller_addr, lm_step_tag=lm_step_tag
    )
    speculative_gen_fn = VLLMSpeculativeRemoteCaller(
        config.LM, config.sLM, config.controller_addr, lm_step_tag=lm_step_tag
    )
    if config.RM == "dummy":
        rm_config = RewardModelBaseConfig(
            step_tag=prm_step_tag, format_str=prm_format_str
        )
        rm_call = DummyRewardModelCaller(rm_config)
    else:
        rm_config = RemoteRewardModelConfig(
            step_tag=prm_step_tag,
            format_str=prm_format_str,
            model_name=config.RM,
            controller_addr=config.controller_addr,
        )
        rm_call = RMRemoteCaller(rm_config)

    speculative_model = None


    # Evaluation Function
    def parallel_evaluate_test_dataset(
        method_name: str, solver_fn, save_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        if save_dir is not None:
            record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w")
        else:
            record_writer = None

        if config.task_name in {'MATH100_qwen', 'MATH100_llama', 'GSM8K100_qwen', 'GSM8K100_llama', 'MATH50_qwen'}:
            task = Task(task_name=config.task_name, is_few_shot=config.is_few_shot)
            evaluator = MathEvaluator(config.task_name, llm_gen_fn, appro_gen_fn, speculative_gen_fn, rm_call, config.method, config.seed, config.c, config.baseline_mode, config.model_mode, config.N, config.M, config.is_tensorboard, config.full_reward, config.x)
            test_ds = task.test_ds
            results = []
            times = []
            # The test dataset is processed in parallel using the actor_pool.map_unordered method, with each problem submitted to a remote evaluator for evaluation
            res_q = list(tqdm(
                map(lambda x: evaluator.evaluate_problem(x[1], x[0], solver_fn), enumerate(test_ds)),
                total=len(test_ds)  
            ))
        else:
            error_message = "Error: dataset is wrong."
            raise ValueError(error_message)  # Throw an exception and terminate the program
            
        if config.task_name in {'MATH100_qwen', 'MATH100_llama', 'GSM8K100_qwen', 'GSM8K100_llama', 'MATH50_qwen'}:
            for i, (problem_inst, result, output, time) in enumerate(
                tqdm(res_q, total=len(test_ds))
            ):
                results.append(result)
                times.append(time)
                if record_writer:
                    obj = {
                        # "i": i,
                        "question": problem_inst["question"],
                        "groundtruth": problem_inst["answer"],
                        "result": result,
                        "output": output,
                        "time": time
                    }
                    # record_writer.write(obj)
        else:
            error_message = "Error: dataset is wrong."
            raise ValueError(error_message)  # Throw an exception and terminate the program
        
        # Calculate the average result
        avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *results),)
        avg_time = [sum(col) / len(col) for col in zip(*times)]
        if record_writer:
            json.dump(avg_res, open(save_dir / "avg_result.json", "w"))
        print("Method: {}. Average result: {}. Time per question: {}"\
              .format(method_name, avg_res, avg_time))
        logging.info(f"Updated Method: {method_name}")
        logging.info(f"Updated Average result: {avg_res}")
        logging.info(f"Updated Time per question: {avg_time}")
        return results

    cfg_dict_record = dict()
    # XXX: qwen-2.5 requires add more stop words
    # not do it now.
    # stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    gen_config = LMCallingConfig(
        n=config.num_sequence,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
    )
    cfg_dict_record["gen_config"] = gen_config.__dict__

    if config.method == "beam_search":
        method_config = BeamSearchConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            beam_size=config.num_sequence,
        )
        solver_fn = partial(beam_search, method_config, gen_config)
    elif config.method == "vanila_mcts":
        method_config = VanilaMCTSConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            select_by_prior=False,
            num_path=config.num_sequence,
        )
        solver_fn = partial(vanila_mcts, method_config, gen_config)
    else:
        raise ValueError(f"Unknown method: {config.method}")
    cfg_dict_record["method"] = config.method
    cfg_dict_record["method_config"] = method_config.__dict__

    if config.save_dir is not None:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(config.save_dir) / config.task_name / config.method / datetime_str
        save_dir.mkdir(parents=True)
        record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w")
        cfg_dict_record["LM"] = config.LM
        cfg_dict_record["RM"] = config.RM
        json.dump(cfg_dict_record, open(save_dir / "config.json", "w"))
    else:
        save_dir = None

    parallel_evaluate_test_dataset(config.method, solver_fn, save_dir)
    total_time = time.time() - start_time
    print(total_time)
    logging.info(f"Updated total time: {total_time}")

