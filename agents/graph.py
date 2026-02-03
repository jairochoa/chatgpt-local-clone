from __future__ import annotations
import json
from typing import TypedDict, List, Literal, Optional
from langgraph.graph import StateGraph, END
import requests
from config.settings import load_config
from agents.tools_pc import list_files, read_text, write_text

cfg = load_config()

Role = Literal["system", "user", "assistant"]

class Msg(TypedDict):
    role: Role
    content: str

class State(TypedDict):
    messages: List[Msg]
    route: Optional[str]          # "chat" | "pc"
    tool_result: Optional[str]    # salida de tool si aplica

def _call_ollama_text(messages: List[Msg]) -> str:
    payload = {
        "model": cfg.ollama.text_model,
        "messages": messages,
        "options": {"temperature": cfg.ollama.temperature},
        "stream": False,
    }
    url = f"{cfg.ollama.base_url}/api/chat"
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]

def route_node(state: State) -> State:
    user_text = (state["messages"][-1]["content"] or "").strip()

    # Primer token (para comandos tipo: list, list ., read hola.txt, write ...)
    first = user_text.split(maxsplit=1)[0].lower() if user_text else ""

    # Si es comando, SIEMPRE va al agente PC
    if first in {"list", "ls", "read", "cat", "write"}:
        state["route"] = "pc"
        return state

    # Router por palabras clave (lenguaje natural)
    t = user_text.lower()
    keywords = [
        "archivo", "archivos", "carpeta", "directorio", "leer", "escribir",
        "guardar", "crear", "listar", "sandbox", "c:\\ai_workspace", "ai_workspace"
    ]
    state["route"] = "pc" if any(k in t for k in keywords) else "chat"
    return state


def chat_agent_node(state: State) -> State:
    # Chat normal (sin tools)
    sys = {"role": "system", "content": "Eres un asistente útil. Responde en español."}
    messages = [sys] + state["messages"]
    answer = _call_ollama_text(messages)
    state["messages"].append({"role": "assistant", "content": answer})
    return state

def pc_agent_node(state: State) -> State:
    user_text = state["messages"][-1]["content"].strip()

    # Si el usuario usa comandos explícitos, mantenemos compatibilidad
    low = user_text.lower()
    if low.startswith(("list", "read ", "write ")):
        # Reusar tu lógica actual (misma que ya funciona)
        try:
            if low.startswith("list"):
                parts = user_text.split(maxsplit=1)
                rel = parts[1].strip().strip('"').strip("'") if len(parts) == 2 else "."
                out = list_files(rel)
                answer = "Archivos:\n- " + "\n- ".join(out) if out else "No hay archivos."
            elif low.startswith("read "):
                rel = user_text[5:].strip().strip('"').strip("'")
                out = read_text(rel)
                answer = f"Contenido de {rel}:\n\n{out}"
            elif low.startswith("write "):
                rest = user_text[6:]
                if " ::: " not in rest:
                    raise ValueError("Formato inválido. Usa: write <archivo> ::: <contenido>")
                rel, content = rest.split(" ::: ", 1)
                rel = rel.strip().strip('"').strip("'")
                result = write_text(rel, content, overwrite=True)
                answer = result
            else:
                answer = "Comando no reconocido."
        except Exception as e:
            answer = f"Error: {e}"

        state["messages"].append({"role": "assistant", "content": answer})
        return state

    # ---- Modo lenguaje natural ----
    # Le pedimos al modelo que nos devuelva una acción JSON
    schema_prompt = (
        "Convierte la petición del usuario en una acción JSON para un agente que SOLO puede operar en un sandbox de archivos.\n"
        "Responde SOLO con JSON válido, sin texto extra. Esquema:\n"
        "{"
        "\"action\": \"list\"|\"read\"|\"write\"|\"ask\","
        "\"path\": \"ruta_relativa_o_nombre\" (opcional),"
        "\"content\": \"texto\" (solo si action=write),"
        "\"question\": \"pregunta\" (solo si action=ask)"
        "}\n\n"
        "Reglas:\n"
        "- Si pide listar archivos => action=list, path opcional (usa '.' si no especifica).\n"
        "- Si pide leer un archivo => action=read, path requerido.\n"
        "- Si pide guardar/crear/escribir => action=write, path y content requeridos.\n"
        "- Si falta información esencial => action=ask y pregunta qué falta.\n"
        "- Nunca inventes contenido de archivos. Nunca uses rutas fuera del sandbox.\n"
    )

    messages = [
        {"role": "system", "content": schema_prompt},
        {"role": "user", "content": user_text},
    ]

    raw = _call_ollama_text(messages)

    # Intentamos parsear JSON
    try:
        data = json.loads(raw)
    except Exception:
        # Fallback: si el modelo responde mal, pedimos aclaración
        answer = (
            "No pude convertir tu petición en una acción segura. "
            "Dime si quieres: listar archivos, leer un archivo, o escribir uno."
        )
        state["messages"].append({"role": "assistant", "content": answer})
        return state

    action = (data.get("action") or "").lower()

    try:
        if action == "list":
            rel = (data.get("path") or ".").strip().strip('"').strip("'")
            out = list_files(rel)
            answer = "Archivos:\n- " + "\n- ".join(out) if out else "No hay archivos."
        elif action == "read":
            rel = (data.get("path") or "").strip().strip('"').strip("'")
            if not rel:
                raise ValueError("Falta 'path' para leer.")
            out = read_text(rel)
            answer = f"Contenido de {rel}:\n\n{out}"
        elif action == "write":
            rel = (data.get("path") or "").strip().strip('"').strip("'")
            content = data.get("content") or ""
            if not rel or not content:
                raise ValueError("Faltan 'path' o 'content' para escribir.")
            result = write_text(rel, content, overwrite=True)
            answer = result
        else:
            q = data.get("question") or "¿Qué quieres hacer con los archivos en el sandbox?"
            answer = q

    except Exception as e:
        answer = f"Error: {e}"

    state["messages"].append({"role": "assistant", "content": answer})
    return state


def build_graph():
    g = StateGraph(State)

    g.add_node("route", route_node)
    g.add_node("chat", chat_agent_node)
    g.add_node("pc", pc_agent_node)

    g.set_entry_point("route")

    def choose(state: State) -> str:
        return state["route"] or "chat"

    g.add_conditional_edges("route", choose, {"chat": "chat", "pc": "pc"})
    g.add_edge("chat", END)
    g.add_edge("pc", END)

    return g.compile()

GRAPH = build_graph()

def run_graph(user_message: str) -> str:
    state: State = {"messages": [{"role": "user", "content": user_message}], "route": None, "tool_result": None}
    out = GRAPH.invoke(state)
    return out["messages"][-1]["content"]
