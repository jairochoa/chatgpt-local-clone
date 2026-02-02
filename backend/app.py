from fastapi import FastAPI, UploadFile, File, Form
import base64
from pydantic import BaseModel
from typing import List, Optional, Literal

from config.settings import load_config
import requests

cfg = load_config()

app = FastAPI(title=cfg.name)

Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class ChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = None

@app.get("/health")
def health():
    return {"status": "ok", "app": cfg.name, "env": cfg.environment}

@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq):
    model = req.model or cfg.ollama.text_model
    temperature = req.temperature if req.temperature is not None else cfg.ollama.temperature

    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "options": {"temperature": temperature},
        "stream": False,
    }

    # Llamada directa a API HTTP de Ollama (simple y explícita)
    url = f"{cfg.ollama.base_url}/api/chat"
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    text = data["message"]["content"]
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }

@app.post("/v1/vision")
async def vision_qa(
    image: UploadFile = File(...),
    prompt: str = Form("Describe la imagen en español con detalle."),
    temperature: Optional[float] = Form(None),
):
    model = cfg.ollama.vision_model
    temp = temperature if temperature is not None else cfg.ollama.temperature

    content = await image.read()
    img_b64 = base64.b64encode(content).decode("utf-8")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }
        ],
        "options": {"temperature": temp},
        "stream": False,
    }

    url = f"{cfg.ollama.base_url}/api/chat"
    r = requests.post(url, json=payload, timeout=240)
    r.raise_for_status()
    data = r.json()

    text = data["message"]["content"]
    return {"model": model, "answer": text}
