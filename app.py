from llama import Llama, AsyncLlama
from pydantic import BaseModel
from typing import Literal, Optional, List
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, Response, StreamingResponse



Role = Literal["system", "user", "assistant"]

app = FastAPI()

class Message(BaseModel):
    role: Role
    content: str


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

# @latency_benchmark
@app.post("/chat_completion")
async def chat_completion(dialog: List[Message] = Body(...)) -> ChatPrediction:
    
    result = generator.chat_completion(
        [[m.dict() for m in dialog]],  # type: ignore
        max_gen_len=1024,
        temperature=0.6,
        top_p=0.9,
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
    
@app.get("/chat_completion_stream")
async def chat_completion_stream(dialog: List[Message] = Body(...), response: StreamingResponse = None):
        
    response = StreamingResponse(generator.chat_completion_async(
            [[m.dict() for m in dialog]],  # type: ignore
            max_gen_len=1024,
            temperature=0.6,
            top_p=0.9,
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
