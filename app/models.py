from pydantic import BaseModel, Field

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
