from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAIError

from app.travel_advisor import chat

app = FastAPI(title="AI Travel Advisor")


class Message(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def send_message(data: Message):
    try:
        if len(data.history) > 30:
            raise HTTPException(400, "History too long")

        cleaned_history = []

        for item in data.history:
            if (
                item.get("role") not in ("user", "assistant")
                or not isinstance(item.get("content"), str)
                or len(item["content"]) > 10000
            ):
                raise HTTPException(400, "Invalid history")

            cleaned_history.append({
                "role": item["role"],
                "content": item["content"]
            })

        answer, history = chat(data.message, cleaned_history)

        return {
            "answer": answer,
            "history": history
        }

    except OpenAIError:
        raise HTTPException(
            status_code=502,
            detail="AI service temporarily unavailable"
        )
from fastapi.responses import FileResponse

@app.get("/")
def homepage():
    return FileResponse("public/index.html")