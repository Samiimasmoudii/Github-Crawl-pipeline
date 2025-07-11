from typing import Generic, TypeVar, Union, NamedTuple, Protocol, Optional, runtime_checkable, Tuple
from abc import ABC, abstractmethod

import numpy as np
from transformers import StoppingCriteriaList
import inspect
from datetime import datetime
import os, sys, pickle
from tqdm import tqdm
import torch
import pickle
from datetime import datetime
import os, sys, pickle
from tqdm import tqdm
import torch

State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = tuple[list[State], list[Action]]


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class GenerateOutput(NamedTuple):
    text: list[str]
    log_prob: Optional[list[np.ndarray]] = None


class LanguageModel(ABC):

    @abstractmethod
    def generate(self,
                 inputs: list[str],
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, int, list[str, int]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 **kwargs) -> GenerateOutput:
        """Generate text from a list of prompts.

        :param inputs: List of prompts.
        :param max_length: Maximum length of the total output (input + generated).
        :param max_new_tokens: Maximum length of generated tokens. Override max_length.
        :param do_sample: If False, do greedy decoding.
        :param temperature: Temperature for sampling.
        :param top_k: Top-k for sampling.
        :param top_p: Top-p for sampling.
        :param num_return_sequences:
        :param eos_token_id: Token id for end of sentence. Passed *str* will be translated into token_id.
                             Passed *list* will be treated as multiple possible tokens ending the generation.
        :param hide_input: If set true, decode only the generated part.
        :param output_log_probs: If set true, also output the log_probs of each generated token
        :param stopping_criteria:
        """
        ...

    @abstractmethod
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              postprocess: Optional[str] = None,
                              **kwargs) -> list[np.ndarray]:
        """ TODO: doc

        :param prompt:
        :param candidates:
        :param postprocess: optional, can be 'log_softmax' or 'softmax'. Apply the corresponding function to logits before returning
        :return:
        """
        ...

    @abstractmethod
    def get_loglikelihood(self, prefix: str, contents: list[str],
                          **kwargs) -> np.ndarray:
        """Get the log likelihood of the contents given the prefix.

        :param prefix: The prefix to be excluded from the log likelihood.
        :param contents: The contents to evaluate (must include the prefix).
        """
        ...


    @abstractmethod
    def reward(self, 
               prefix: str, 
               contents: list[str], 
               reward_model,
               **kwargs) -> np.ndarray:
        ...


class WorldModel(ABC, Generic[State, Action, Example]):

    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def init_state(self) -> State:
        ...

    @abstractmethod
    def step(self, state: State,
             action: Action) -> Union[State, Tuple[State, dict]]:
        """ Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        ...

    def update_example(self, example: Example, prompt=None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example


class DefaultWorldModel(WorldModel):
    # A default implementation of WorldModel that only
    # saves the action sequence as the state

    def __init__(self, base_model) -> None:
        super().__init__()
        self.base_model = base_model

    def init_state(self):
        return []

    def step(self, state, action):
        return state + [action], {}

    def is_terminal(self, state):
        # By default the state is never terminal
        return False


class SearchConfig(ABC, Generic[State, Action, Example]):

    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]:
        ...

    def fast_reward(self, state: State, action: Action) -> tuple[float, dict]:
        return 0, {}

    @abstractmethod
    def reward(self, 
               state, 
               action, 
               reward_model, 
               **kwargs) -> tuple[float, dict]:
        ...

    def update_example(self, example: Example, prompt=None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example


@runtime_checkable
class AlgorithmOutput(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):

    def __init__(self, **kwargs):
        ...

    @abstractmethod
    def __call__(self, world_model: WorldModel, search_config: SearchConfig,
                 **kwargs) -> AlgorithmOutput:
        ...


class Reasoner(ABC, Generic[State, Action, Example]):

    def __init__(self, world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 search_algo: SearchAlgorithm) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(self,
                 example: Example,
                 prompt=None,
                 **kwargs) -> AlgorithmOutput[State]:
        self.world_model.update_example(example, prompt=prompt)
        self.search_config.update_example(example, prompt=prompt)

        return self.search_algo(self.world_model, self.search_config, **kwargs)
    


class Evaluator():

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def sample_prompt(self, shuffle_prompt, num_shot, sample_prompt_type):
        pass

    def evaluate(self,
                 reasoner,
                 shuffle_prompt=True,
                 num_shot=4,
                 start_idx=0,
                 end_idx=0,
                 log_dir=None,
                 pickle_path=None):
        
        if start_idx is None:
            start_idx = 0
        elif start_idx > 0:
            print(f"resume at: {start_idx}")

        print()
        self.dataset = list(self.full_dataset)[start_idx:]
        correct_count = 0

        for i, example in enumerate(tqdm(self.dataset,
                                    total=len(self.dataset),
                                    initial=start_idx,
                                    desc=self._dataset_name,
                                    disable=False)):
            
            # try:
            print(f"\ndata idx: {i}")
            n_shot_prompt = self.sample_prompt(shuffle_prompt=shuffle_prompt,
                                               num_shot=num_shot)

            question_prompt = self.input_processor(example)

            algo_output = reasoner(question_prompt, n_shot_prompt)

            if pickle_path is not None:
                dir_name = os.path.dirname(f"{pickle_path}_{i}.pkl")
    
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                    
                with open(f"{pickle_path}_{i}.pkl", 'wb') as f:
                    pickle.dump(algo_output, f)            

            output = self.output_extractor(algo_output)

            print("\n\n"+"="*60+"evaluate start"+"="*60)

            answer = self.answer_extractor(example)

            correct = self.eval_output(answer, output)

            print(f"answer: {answer}")
            print(f"output: {output}")
            print(f"correct: {correct}")

            correct_count += correct

            accuracy = correct_count / (i + 1)

            print("="*60+"evaluate end"+"="*60+"\n\n")

            print(f"*****current accuracy: {accuracy}*****\n")
            
            # except Exception as e:
            #     print(f"error at index: {i}: {e}")

            # with open("log/gsm8k/cot/llama-3-8b-gptq.log", 'a') as f:
            #     f.write('=' * 200 + '\n')
            #     f.write(f'Index: {i}\n')
            #     f.write(f'Question: {question_prompt}\n')
            #     f.write(f'Model output: {algo_output}\n')
            #     f.write(f'Extracted model answer: {output}\n')
            #     f.write(f'Standard answer: {answer}\n')
            #     f.write(f'Correct: {correct}\n')

        return accuracy

    def evaluate_sample_chain(self,
                              reasoner,
                              shuffle_prompt=True,
                              num_shot=4,
                              resume=0,
                              num_sample_chain=10,
                              log_dir=None):

        self.dataset = list(self.full_dataset)[resume:]

        correct_count = 0

        for i, example in enumerate(
                tqdm(self.dataset,
                     total=resume + len(self.dataset),
                     initial=resume,
                     desc=self._dataset_name,
                     disable=self.disable_tqdm)):

            prompt = self.sample_prompt(shuffle_prompt=shuffle_prompt,
                                        num_shot=num_shot)
            output_list = []
            save_list = []

            for _ in range(num_sample_chain):
                algo_output = reasoner(self.input_processor(example),
                                       prompt=prompt)
                terminal_state = algo_output.terminal_state
                path = ""
                for k in range(len(terminal_state)):
                    path += terminal_state[
                        k].sub_question + " " + terminal_state[
                            k].sub_answer + " "
                save_list.append(path)
                output = self.output_extractor(algo_output)
                output_list.append(output)
                answer = self.answer_extractor(example)

            from collections import Counter
            output = Counter(output_list).most_common(1)[0][0]
            correct = self.eval_output(answer, output)
            correct_count += correct
            accuracy = correct_count / (i + 1)

            # with open("log/gsm8k/cot/llama-3-8b-gptq.log", 'a') as f:
            #     f.write('=' * 200 + '\n')
            #     f.write(f'Index: {i}\n')
            #     f.write(f'Question: {question_prompt}\n')
            #     f.write(f'Model output: {algo_output}\n')
            #     f.write(f'Extracted model answer: {output}\n')
            #     f.write(f'Standard answer: {answer}\n')
            #     f.write(f'Correct: {correct}\n')

        return accuracy

    @abstractmethod
    def eval_output(self, answer, output):
        pass


class Tool():

    def __init__(self, func, name, description):
        self.func = func
        self.name = name
        self.description = description

    def __call__(self, **kwargs):
        return self.func(**kwargs)
