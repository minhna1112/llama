# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
import torch
import torch.nn.functional as F

sys.path.append(".")
from tokenizer import Tokenizer
from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as client_util
import tritonclient.grpc.aio as grpcclient

sys.path.append("..")

from decorators import latency_benchmark

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def prepare_tensor(name, input):
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


class LlamaTritonClient:
    

    def __init__(self, model_name,
                 host='localhost',
                 port=8001,
                 verbose=False,
                 grpc_keepalive_timeout=20000):
        
        self.url = f'{host}:{port}'
        self.verbose = verbose
        self.model_name = model_name
        self.keepalive_options = client_util.KeepAliveOptions(keepalive_timeout_ms=grpc_keepalive_timeout) #by defaul: 20s
        self.client = client_util.InferenceServerClient(self.url,
                                                        verbose=self.verbose,
                                                        keepalive_options=self.keepalive_options)
        #self.triton_client = grpcclient.InferenceServerClient(self.url,verbose=self.verbose)
        self.max_batch_size = 4
        self.tokenizer  = Tokenizer(model_path="../tokenizer.model")

    def server_sync_generate(self,
            tokens: torch.Tensor, 
            min_prompt_len,
            total_len,
            temperature,
            top_p,
            )-> Tuple[np.ndarray, np.ndarray]:
        
        bsz = tokens.size(0)
        inputs = [
                    prepare_tensor("input_ids", tokens.numpy().astype(np.int64)),
                    prepare_tensor("min_prompt_len", min_prompt_len * np.ones(shape=(bsz,1), dtype=np.int32)),
                    prepare_tensor("total_len", total_len * np.ones(shape=(bsz,1), dtype=np.int32)),
                    prepare_tensor("temperature", temperature * np.ones(shape=(bsz,1), dtype=np.float32)),
                    prepare_tensor("top_p", top_p * np.ones(shape=(bsz,1), dtype=np.float32))
            ]
            
        result = self.client.infer(self.model_name, inputs)
        output_tokens = result.as_numpy("output_ids")
        output_logprobs = result.as_numpy("log_probs")
        return output_tokens, output_logprobs
        
    def generate(self, 
                 prompt_tokens: List[List[int]],
                 max_gen_len : int = 2048,
                 temperature: int = 0,
                 top_p: int  = 1,
                 logprobs : bool = True,
                 echo: bool = False,
                 max_seq_len : int =4096
                 )->Tuple[List]:
        bsz = len(prompt_tokens)
        assert bsz <= self.max_batch_size
        
        if not max_gen_len:
            max_gen_len =  max_prompt_len
        
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        print("max input length in a batch", max_prompt_len)
        assert max_prompt_len <= max_seq_len
        total_len = min(max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        # place-holder for input and output tokens
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        # Inference at server-side
        tokens, token_logprobs = self.server_sync_generate(
            tokens, 
            min_prompt_len,
            total_len,
            temperature,
            top_p,
            # logprobs
        )
        
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
    
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    @latency_benchmark
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        
        prompt_tokens = []
        for dialog in dialogs:
            if dialog[0]["role"] != "system":
                dialog = [
                    {
                        "role": "system",
                        "content": DEFAULT_SYSTEM_PROMPT,
                    }
                ] + dialog
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        
        assert len(prompt_tokens) <= self.max_batch_size
        
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        
        
        
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [
            {"generation": {"role": "assistant", "content": self.tokenizer.decode(t)}}
            for t in generation_tokens
        ]


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

if __name__=='__main__':
    client  = LlamaTritonClient(
        "torch",
        host="localhost",
        port=8010
    )
    out = client.chat_completion(
        [
            # Dialog(
                [
                    {
                    "content": "You are a coding assistant named LLamaCoding",
                    "role": "system" 
                    },                
                    {
                        "content": "What is your name?",
                        "role": "user"
                    }
                ]               
            # ) 
            for i in range(4)
        ],
        max_gen_len=2048,
        # logprobs=True
    )
    print(out)