from typing import List, Literal, Optional, Tuple, TypedDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import torch
from threading import Thread
import json

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

USER_PREFIX = "### Instruction\n"
ASSISTANT_PREFIX = "### Response\n"

class StableCode:
    def __init__(self, checkpoint_path: str = "/stablecode-instruct-alpha-3b") -> None:
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        print("Loaded")
        if torch.cuda.is_available():
            print("Moving model to GPU")
            self.model = self.model.to("cuda")
            print("Model moved to GPU")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print("Loaded")
    
    async def chat_completion_async(       
            self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False):
        
        input_text  = "\n".join([
                    USER_PREFIX + msg["content"] 
                        if msg["role"] in ["user", "system"] 
                            else ASSISTANT_PREFIX + msg["content"] 
                    for msg in dialogs[-1]
                ])+"\n"+ASSISTANT_PREFIX
        
        print(input_text)
        
        inputs = self.tokenizer(
            [
                input_text
            ]
            , return_tensors="pt"
        )
        
        inputs.pop("token_type_ids")
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        print(inputs["input_ids"].size())
        
        
        generation_kwargs = dict(inputs, 
                                 streamer=streamer, 
                                 max_new_tokens=max_gen_len,
                                 top_p=top_p,
                                 temperature = temperature,
                                 do_sample=True)
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        generated = ""
        
        for generated_text in streamer:
            # generated += generated_text
            print(generated_text)
            yield json.dumps(
                    {
                        "generation":
                            {
                                "role": "assistant",
                                "content": generated_text
                            }
                    }
                )
        
        
        
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ):
        """Complete a dialog by generating a response.

        Args:
            dialogs: List of dialogs as lists of messages
            temperature: Sampling temperature
            top_p: Nucleus sampling top-p
            max_gen_len: Max length of generated response
            logprobs: Whether to return log probabilities
        """
        input_text  = "\n".join([
                    USER_PREFIX + msg["content"] 
                        if msg["role"] in ["user", "system"] 
                            else ASSISTANT_PREFIX + msg["content"] 
                    for msg in dialogs[-1]
                ])+"\n"+ASSISTANT_PREFIX
        
        print(input_text)
        
        inputs = self.tokenizer(
            [
                input_text
            ]
            , return_tensors="pt"
        )
        
        print(inputs["input_ids"].size())
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        generated = self.model.generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        # generated[inputs["attention_mask"]] = torch.full(inputs["attention_mask"].size(), self.tokenizer.pad_token_id).to("cuda")
        generated = generated.detach().cpu()
        # generated[generated==inputs["input_ids"].detach().cpu()] = self.tokenizer.pad_token_id

        response = self.tokenizer.decode(generated[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        print(response)
        
        # if logprobs:
        #     logprobs = self.model.get_logits(generated).tolist()
        
        return {
                "generation":
                    {
                        "role": "assistant",
                        "content": response
                    }
                }