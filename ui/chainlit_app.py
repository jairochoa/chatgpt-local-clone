import chainlit as cl
import httpx
from config.settings import load_config

cfg = load_config()
CHAT_ENDPOINT = f"http://{cfg.backend.host}:{cfg.backend.port}/v1/chat/completions"
VISION_ENDPOINT = f"http://{cfg.backend.host}:{cfg.backend.port}/v1/vision"

@cl.on_message
async def on_message(message: cl.Message):
    # 1) Intentar visión si hay imagen
    img = None
    if message.elements:
        for el in message.elements:
            mime = getattr(el, "mime", "") or ""
            path = str(getattr(el, "path", "") or "").lower()
            if mime.startswith("image/") or path.endswith((".png", ".jpg", ".jpeg", ".webp")):
                img = el
                break

    async with httpx.AsyncClient(timeout=300) as client:
        if img:
            file_bytes = None

            if getattr(img, "content", None):
                file_bytes = img.content
            elif getattr(img, "path", None):
                with open(img.path, "rb") as f:
                    file_bytes = f.read()

            if not file_bytes:
                await cl.Message(content="No pude leer la imagen (sin path ni content).").send()
                return

            files = {"image": ("image", file_bytes, "application/octet-stream")}
            data = {"prompt": message.content or "Describe la imagen en español."}

            r = await client.post(VISION_ENDPOINT, files=files, data=data)
            r.raise_for_status()
            answer = r.json().get("answer", "Sin respuesta.")
            await cl.Message(content=answer).send()
            return

        # 2) Chat normal
        payload = {
            "messages": [
                {"role": "system", "content": "Eres un asistente útil. Responde en español."},
                {"role": "user", "content": message.content},
            ]
        }
        r = await client.post(CHAT_ENDPOINT, json=payload)
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"]
        await cl.Message(content=answer).send()
