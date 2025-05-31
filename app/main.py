from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.auth import authenticate
from app.models import PromptRequest
from app.llm import generate_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.post("/generate")
async def generate(prompt_req: PromptRequest, _ = Depends(authenticate)):
    result = generate_response(
        prompt_req.prompt,
        prompt_req.max_tokens,
        prompt_req.temperature
    )
    return {"response": result}
