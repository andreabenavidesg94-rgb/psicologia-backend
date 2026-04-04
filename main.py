import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./psicologia.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class MemoryItem(Base):
    _tablename_ = "memory_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True, nullable=False)
    kind = Column(String(50), nullable=False)  # profile, journal, summary, topic
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ChatTurn(Base):
    _tablename_ = "chat_turns"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True, nullable=False)
    role = Column(String(20), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class JournalEntry(Base):
    _tablename_ = "journal_entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True, nullable=False)
    title = Column(String(255), nullable=True)
    text = Column(Text, nullable=False)
    emotion = Column(String(50), nullable=True)
    intensity = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


app = FastAPI(title="PsicologIA PRO Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    mensaje: str
    user_name: str | None = "Tú"
    assistant_name: str | None = "Andrea"
    focus_area: str | None = "Ansiedad"
    remembered_topic: str | None = ""
    user_id: str | None = None


class DiarioRequest(BaseModel):
    texto: str
    title: str | None = None
    user_name: str | None = "Tú"
    assistant_name: str | None = "Andrea"
    focus_area: str | None = "Ansiedad"
    user_id: str | None = None


def now_utc():
    return datetime.now(timezone.utc)


def normalize_user_id(user_id: Optional[str], user_name: Optional[str]) -> str:
    if user_id and user_id.strip():
        return user_id.strip()
    if user_name and user_name.strip():
        return f"user::{user_name.strip().lower()}"
    return "user::anonimo"


def save_chat_turn(db, user_id: str, role: str, content: str):
    db.add(ChatTurn(user_id=user_id, role=role, content=content))
    db.commit()


def save_memory(db, user_id: str, kind: str, content: str):
    db.add(MemoryItem(user_id=user_id, kind=kind, content=content))
    db.commit()


def save_journal(db, user_id: str, title: str, text: str, emotion: str, intensity: int):
    db.add(
        JournalEntry(
            user_id=user_id,
            title=title,
            text=text,
            emotion=emotion,
            intensity=intensity,
        )
    )
    db.commit()


def get_recent_chat(db, user_id: str, limit: int = 8):
    rows = (
        db.query(ChatTurn)
        .filter(ChatTurn.user_id == user_id)
        .order_by(ChatTurn.created_at.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()
    return rows


def get_recent_memories(db, user_id: str, limit: int = 5):
    rows = (
        db.query(MemoryItem)
        .filter(MemoryItem.user_id == user_id)
        .order_by(MemoryItem.created_at.desc())
        .limit(limit)
        .all()
    )
    return rows


def get_recent_journal(db, user_id: str, limit: int = 5):
    rows = (
        db.query(JournalEntry)
        .filter(JournalEntry.user_id == user_id)
        .order_by(JournalEntry.created_at.desc())
        .limit(limit)
        .all()
    )
    return rows


def heuristic_memory_update(message: str) -> Optional[str]:
    lower = message.lower()

    if any(x in lower for x in ["ansiedad", "nervios", "ataque"]):
        return "La ansiedad aparece con frecuencia y conviene trabajarla con más calma y regulación."
    if any(x in lower for x in ["pareja", "relación", "amor"]):
        return "Hay algo importante en el ámbito relacional que le está afectando emocionalmente."
    if any(x in lower for x in ["agot", "cans", "burnout"]):
        return "Se percibe cansancio emocional y necesidad de descanso real."
    if any(x in lower for x in ["solo", "sola"]):
        return "A veces se siente sola/o y necesita más contención emocional."
    if any(x in lower for x in ["triste", "vacío", "llorar"]):
        return "Hay tristeza importante presente en varios momentos."
    return None


def fallback_chat_reply(req: ChatRequest) -> str:
    texto = req.mensaje.lower()

    if "ansiedad" in texto or "nervios" in texto:
        return f"{req.user_name}, gracias por contármelo. Vamos a bajar un poco el ritmo. ¿Qué fue lo que más activó esa ansiedad hoy?"
    if "relación" in texto or "pareja" in texto or "amor" in texto:
        return "Entiendo. Cuando una relación pesa, todo se siente más intenso. ¿Qué fue lo que más te dolió exactamente?"
    if "agot" in texto or "cans" in texto:
        return "Suena a que vienes sosteniendo demasiado. ¿Sientes más cansancio físico, mental o emocional?"
    if "sola" in texto or "solo" in texto:
        return "Estoy aquí contigo. No tienes que cargar con todo sin apoyo. ¿Qué es lo que más pesa ahora mismo?"
    if "triste" in texto or "vacío" in texto:
        return "Gracias por compartirlo conmigo. No voy a minimizar lo que sientes. ¿Desde cuándo notas esta tristeza tan presente?"
    if "hola" in texto or "buenas" in texto:
        return f"Hola, {req.user_name}. Me alegra verte aquí. ¿Cómo te estás sintiendo hoy, de verdad?"
    return f"Te escucho, {req.user_name}. Cuéntame qué parte de esto te gustaría entender mejor."


def build_system_prompt(req: ChatRequest, memory_lines: list[str], journal_lines: list[str]) -> str:
    memories = "\n".join(f"- {m}" for m in memory_lines) if memory_lines else "- Sin memoria previa relevante."
    journals = "\n".join(f"- {j}" for j in journal_lines) if journal_lines else "- Sin registros recientes."

    return f"""
Eres {req.assistant_name}, una asistente emocional cálida, elegante, útil y humana para una app llamada PsicologIA.
Hablas en español.
Tu tono debe sentirse cercano, suave, reconfortante y profesional.
No digas que eres terapeuta ni sustituyes atención profesional, salvo que sea necesario por seguridad.
No seas fría ni demasiado larga.
Responde con 2 a 5 frases.
Haz seguimiento emocional real y continuidad.

Datos del usuario:
- Nombre: {req.user_name}
- Área principal: {req.focus_area}

Memoria emocional relevante:
{memories}

Últimos registros de diario:
{journals}

Si detectas riesgo grave, autolesión o suicidio:
- prioriza seguridad
- recomienda ayuda inmediata
- sugiere contactar emergencias o una persona de confianza

Tu objetivo:
- hacer que el usuario se sienta escuchado
- recordar lo importante
- continuar el proceso emocional
- dar contención y una siguiente pregunta útil
""".strip()


def openai_chat_reply(req: ChatRequest, memory_lines: list[str], journal_lines: list[str], recent_chat_lines: list[str]) -> str:
    if client is None:
        return fallback_chat_reply(req)

    system_prompt = build_system_prompt(req, memory_lines, journal_lines)
    transcript = "\n".join(recent_chat_lines[-8:]) if recent_chat_lines else "Sin conversación reciente."

    user_prompt = f"""
Conversación reciente:
{transcript}

Mensaje actual del usuario:
{req.mensaje}
""".strip()

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (response.output_text or "").strip()
        return text if text else fallback_chat_reply(req)
    except Exception:
        return fallback_chat_reply(req)


def fallback_diary_analysis(texto: str):
    lower = texto.lower()

    emocion = "Calma"
    intensidad = 5
    resumen = "Tu registro quedó guardado correctamente."
    consejo = "Vuelve cuando quieras. Lo importante aquí es seguir conectando contigo."

    if "ansiedad" in lower:
        emocion = "Ansiedad"
        intensidad = 8
        resumen = "Hoy aparece ansiedad como emoción principal."
        consejo = "Prueba bajar el ritmo, reducir estímulos y volver a la respiración."
    elif "triste" in lower:
        emocion = "Tristeza"
        intensidad = 7
        resumen = "Hay una tristeza importante en lo que escribiste."
        consejo = "No intentes resolverlo todo ahora. Dale espacio a lo que sientes."
    elif "agot" in lower or "cans" in lower:
        emocion = "Agotamiento"
        intensidad = 8
        resumen = "Se nota mucho cansancio emocional en este registro."
        consejo = "Tu cuerpo y tu mente parecen estar pidiendo descanso real."
    elif "esperanza" in lower or "mejor" in lower:
        emocion = "Esperanza"
        intensidad = 4
        resumen = "También hay señales de alivio y esperanza en tu escritura."
        consejo = "Vale la pena reconocer lo que sí está mejorando, aunque sea poco."

    return {
        "emocion": emocion,
        "intensidad": intensidad,
        "resumen": resumen,
        "consejo": consejo,
    }


def openai_diary_analysis(texto: str, user_name: str, focus_area: str):
    if client is None:
        return fallback_diary_analysis(texto)

    prompt = f"""
Analiza este diario emocional y devuelve SOLO JSON válido con este formato exacto:
{{
  "emocion": "Calma|Ansiedad|Tristeza|Esperanza|Agotamiento|Confusión",
  "intensidad": 1-10,
  "resumen": "máximo 20 palabras",
  "consejo": "máximo 30 palabras"
}}

Nombre: {user_name}
Área principal: {focus_area}

Texto:
{texto}
""".strip()

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": "Eres una IA que analiza diarios emocionales y responde solo JSON válido en español.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw = (response.output_text or "").strip()
        data = json.loads(raw)
        return {
            "emocion": str(data.get("emocion", "Calma")),
            "intensidad": int(data.get("intensidad", 5)),
            "resumen": str(data.get("resumen", "Tu registro quedó guardado correctamente.")),
            "consejo": str(data.get("consejo", "Vuelve cuando quieras. Lo importante aquí es seguir conectando contigo.")),
        }
    except Exception:
        return fallback_diary_analysis(texto)


@app.get("/")
def root():
    return {"ok": True, "message": "Backend PsicologIA PRO funcionando"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_configured": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
        "database_url_set": bool(DATABASE_URL),
    }


@app.post("/chat")
def chat(req: ChatRequest):
    db = SessionLocal()
    try:
        user_id = normalize_user_id(req.user_id, req.user_name)

        recent_chat = get_recent_chat(db, user_id, limit=8)
        memory_rows = get_recent_memories(db, user_id, limit=5)
        journal_rows = get_recent_journal(db, user_id, limit=5)

        memory_lines = [m.content for m in memory_rows]
        journal_lines = [
            f"{j.created_at.strftime('%d/%m')}: emoción={j.emotion or 'Sin clasificar'} intensidad={j.intensity or 5}/10 texto={j.text[:180]}"
            for j in journal_rows
        ]
        recent_chat_lines = [f"{row.role}: {row.content}" for row in recent_chat]

        save_chat_turn(db, user_id, "user", req.mensaje)

        respuesta = openai_chat_reply(req, memory_lines, journal_lines, recent_chat_lines)

        save_chat_turn(db, user_id, "assistant", respuesta)

        inferred_memory = heuristic_memory_update(req.mensaje)
        if inferred_memory:
            save_memory(db, user_id, "topic", inferred_memory)

        return {
            "respuesta": respuesta,
            "memoria": inferred_memory or (memory_lines[0] if memory_lines else req.remembered_topic or ""),
        }
    finally:
        db.close()


@app.post("/diario")
def diario(req: DiarioRequest):
    db = SessionLocal()
    try:
        user_id = normalize_user_id(req.user_id, req.user_name)
        analysis = openai_diary_analysis(req.texto, req.user_name or "Tú", req.focus_area or "Ansiedad")

        save_journal(
            db,
            user_id=user_id,
            title=req.title or "Mi registro",
            text=req.texto,
            emotion=analysis["emocion"],
            intensity=int(analysis["intensidad"]),
        )

        save_memory(
            db,
            user_id=user_id,
            kind="journal",
            content=f"En el diario reciente aparece {analysis['emocion'].lower()} con intensidad {analysis['intensidad']}/10.",
        )

        return analysis
    finally:
        db.close()