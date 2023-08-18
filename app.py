from llama import Llama, AsyncLlama
from stable_code import StableCode
from vicuna import Vicuna
from pydantic import BaseModel
from typing import Literal, Optional, List
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, Response, StreamingResponse

import os


Role = Literal["system", "user", "assistant"]

app = FastAPI()


class Message(BaseModel):
    role: Role
    content: str

class ModelParams(BaseModel):
    temperature: float = 0
    top_p: float = 1.0
    max_gen_len: int = 1025

class ModelInputs(BaseModel):
    dialog: List[Message]
    model_params: ModelParams 

class InputRequest(ModelInputs):
    model_name: str = "llama"
    

class CompletionPrediction(BaseModel):
    generation: str
    tokens: Optional[List[str]]  # not required
    logprobs: Optional[List[float]]  # not required


class ChatPrediction(BaseModel):
    generation: Message
    tokens: Optional[List[str]]  # not required
    logprobs: Optional[List[float]]  # not required

generator = AsyncLlama.build(
    ckpt_dir="llama-2-7b-chat",
    tokenizer_path="./tokenizer.model",
    max_seq_len=2048,
    max_batch_size=2,
    model_parallel_size=1
)

stablecode = StableCode(
    checkpoint_path="./tritonserver/stable-code/stable-code-3b/1/checkpoints/stablecode-instruct-alpha-3b",
    
)

# vicuna = Vicuna(
#     checkpoint_path="./tritonserver/vicuna-v1.5/vicuna-v1.5-7b/1/checkpoints/vicuna-7b-v1.5-16k",
# )

vicuna = None
# stablecode = None

@app.post("/chat_completion")
async def chat_completion(model_input: InputRequest) -> ChatPrediction:
    model_name = model_input.model_name
    
    if model_name == "llama":
        result = await llama_chat_completion(model_input)
    if model_name == "stablecode":
        result = await stablecode_chat_completion(model_input)
    if model_name == "vicuna":
        result = await vicuna_chat_completion(model_input)
    
    return result

@app.get("/chat_completion_stream")
async def chat_completion(model_input: InputRequest):
    model_name = model_input.model_name
    
    if model_name == "llama":
        result = await llama_chat_completion_stream(model_input)
    if model_name == "stablecode":
        result = await stablecode_chat_completion_stream(model_input)
    if model_name == "vicuna":
        result = await vicuna_chat_completion_stream(model_input)
    return result


# @latency_benchmark
@app.post("/llama/chat_completion")
async def llama_chat_completion(model_input : ModelInputs) -> ChatPrediction:
    # model_input = model_input.dict()
    dialog = model_input.dialog
    print(dialog)
    model_params = model_input.model_params.dict()
    result = generator.chat_completion(
        [[m.dict() for m in dialog]],  # type: ignore
        **model_params
    )[0]

    for msg in dialog:
        print(f"{msg.role.capitalize()}: {msg.content}\n")
    print(
        f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    )
    print("\n==================================\n")

    return ChatPrediction(
        generation=Message(**result["generation"]),
        tokens= result.get("tokens"),
        logprobs=result.get("logprobs")
    )
    
@app.get("/llama/chat_completion_stream")
async def llama_chat_completion_stream(model_input : ModelInputs, response: StreamingResponse = None):
    # model_input = model_input.dict()
    dialog = model_input.dialog
    model_params = model_input.model_params.dict()    
    response = StreamingResponse(generator.chat_completion_async(
        [[m.dict() for m in dialog]],  # type: ignore
        **model_params
        ), media_type="text/event-stream")
    # print(response.body_iterator)
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response        

@app.post("/stablecode/chat_completion")
async def stablecode_chat_completion(model_input: ModelInputs) -> ChatPrediction:
    # model_input = model_input.dict()
    dialog = model_input.dialog
    model_params = model_input.model_params.dict()
    result = stablecode.chat_completion(
        [[m.dict() for m in dialog]],  # type: ignore
        **model_params
    )

    return ChatPrediction(
        generation=Message(**result["generation"]),
        tokens= result.get("tokens"),
        logprobs=result.get("logprobs")
    )

@app.get("/stablecode/chat_completion_stream")
async def stablecode_chat_completion_stream(model_input: ModelInputs):
    # model_input = model_input.dict()
    dialog = model_input.dialog
    model_params = model_input.model_params.dict()
    response = StreamingResponse(stablecode.chat_completion_async(
        [[m.dict() for m in dialog]],  # type: ignore
        **model_params
        ), media_type="text/event-stream")
    # print(response.body_iterator)
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response      


@app.post("/vicuna/chat_completion")
async def vicuna_chat_completion(model_input: ModelInputs) -> ChatPrediction:
    # model_input = model_input.dict()
    dialog = model_input.dialog
    model_params = model_input.model_params.dict()
    result = vicuna.chat_completion(
        [[m.dict() for m in dialog]],  # type: ignore
        **model_params
    )

    return ChatPrediction(
        generation=Message(**result["generation"]),
        tokens= result.get("tokens"),
        logprobs=result.get("logprobs")
    )

@app.get("/vicuna/chat_completion_stream")
async def vicuna_chat_completion_stream(model_input: ModelInputs):
    # model_input = model_input.dict()
    dialog = model_input.dialog
    model_params = model_input.model_params.dict()
    response = StreamingResponse(vicuna.chat_completion_async(
        [[m.dict() for m in dialog]],  # type: ignore
        **model_params
        ), media_type="text/event-stream")
    # print(response.body_iterator)
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response      



@app.exception_handler(Exception)
async def exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
