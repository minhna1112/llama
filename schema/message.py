from pydantic import BaseModel
from typing import List, Literal, Optional, Tuple, TypedDict, Union

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str

class ChatPrediction(BaseModel):
    generation: Message
    tokens: Optional[List[str]]  # not required
    logprobs: Optional[List[float]]  # not required

class Choice(BaseModel):
    index: int
    message: Optional[Message]
    finish_reason: Optional[Literal["stop", "out_of_tokens"]]

# class Text()

class Delta(BaseModel):
    content: str

class StreamChoice(Choice):
    delta: Delta

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class LLMChatOutput(BaseModel):
    id: str
    created: int
    model : Optional[str]
    choices: List[Union[Choice,  StreamChoice]]
    usage: Optional[TokenUsage]
    