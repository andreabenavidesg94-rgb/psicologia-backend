import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from openai import OpenAI

import firebase_admin
from firebase_admin import auth, credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./psicologia.db")

FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
GOOGLE_PLAY_SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_PLAY_SERVICE_ACCOUNT_PATH", "").strip()
ANDROID_PACKAGE_NAME = os.getenv("ANDROID_PACKAGE_NAME", "com.albe.bienestaria").strip()

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def init_firebase() -> None:
    if firebase_admin._apps:
        return
    if not FIREBASE_SERVICE_ACCOUNT_PATH:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_PATH no configurado")
    cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)


def build_android_publisher():
    if not GOOGLE_PLAY_SERVICE_ACCOUNT_PATH:
        raise RuntimeError("GOOGLE_PLAY_SERVICE_ACCOUNT_PATH no configurado")

    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_PLAY_SERVICE_ACCOUNT_PATH,
        scopes=["https://www.googleapis.com/auth/androidpublisher"],
    )
    return build("androidpublisher", "v3", credentials=creds, cache_discovery=False)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_google_rfc3339(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def verify_firebase_token(authorization: Optional[str]) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Falta Authorization Bearer token")
    token = authorization.replace("Bearer ", "", 1).strip()
    if not token:
        raise HTTPException(status_code=401, detail="Firebase token vacío")

    try:
        return auth.verify_id_token(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Firebase token inválido: {str(e)}")


class MemoryItem(Base):
    __tablename__ = "memory_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True, nullable=False)
    kind = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=now_utc)


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True, nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=now_utc)


class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True, nullable=False)
    title = Column(String(255), nullable=True)
    text = Column(Text, nullable=False)
    emotion = Column(String(50), nullable=True)
    intensity = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=now_utc)


class SubscriptionEntitlement(Base):
    __tablename__ = "subscription_entitlements"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True, nullable=False)
    firebase_uid = Column(String(255), index=True, nullable=False)
    email = Column(String(255), nullable=True)
    product_id = Column(String(255), nullable=False)
    purchase_token = Column(Text, nullable=False, unique=True)
    plan = Column(String(50), nullable=False, default="free")
    source = Column(String(50), nullable=False, default="google_play")
    status = Column(String(50), nullable=False, default="inactive")
    is_active = Column(Boolean, nullable=False, default=False)
    expiry_date = Column(DateTime, nullable=True)
    latest_order_id = Column(String(255), nullable=True)
    raw_payload = Column(Text, nullable=True)
    created_at = Column(DateTime, default=now_utc)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc)


Base.metadata.create_all(bind=engine)

app = FastAPI(title="PsicologIA PRO Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    init_firebase()


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


class VerifySubscriptionRequest(BaseModel):
    purchase_token: str
    product_id: str
    package_name: Optional[str] = None


class RestoreSubscriptionRequest(BaseModel):
    purchase_tokens: list[str]
    product_id_hint: Optional[str] = None
    package_name: Optional[str] = None


def normalize_user_id(user_id: Optional[str], user_name: Optional[str]) -> str:
    if user_id and user_id.strip():
        return user_id.strip()
    if user_name and user_name.strip():
        return f"user::{user_name.strip().lower()}"
    return "user::anonimo"


def map_plan_from_product_id(product_id: str) -> str:
    product_id = product_id.lower()
    if "premium" in product_id:
        return "premium"
    if "plus" in product_id:
        return "plus"
    return "free"


def map_status_from_google_payload(payload: dict[str, Any]) -> tuple[str, bool]:
    state = (payload.get("subscriptionState") or "").upper()

    if state == "SUBSCRIPTION_STATE_ACTIVE":
        return "active", True
    if state == "SUBSCRIPTION_STATE_IN_GRACE_PERIOD":
        return "active", True
    if state == "SUBSCRIPTION_STATE_ON_HOLD":
        return "on_hold", False
    if state == "SUBSCRIPTION_STATE_PAUSED":
        return "paused", False
    if state == "SUBSCRIPTION_STATE_CANCELED":
        return "canceled", False
    if state == "SUBSCRIPTION_STATE_EXPIRED":
        return "expired", False
    if state == "SUBSCRIPTION_STATE_PENDING":
        return "pending", False
    return "inactive", False


def extract_expiry_date(payload: dict[str, Any]) -> Optional[datetime]:
    line_items = payload.get("lineItems") or []
    expiries: list[datetime] = []
    for item in line_items:
        dt = parse_google_rfc3339(item.get("expiryTime"))
        if dt:
            expiries.append(dt)
    return max(expiries) if expiries else None


def extract_order_id(payload: dict[str, Any]) -> Optional[str]:
    line_items = payload.get("lineItems") or []
    for item in line_items:
        latest = item.get("latest_successful_order_id")
        if latest:
            return str(latest)
    return None


def google_verify_subscription(package_name: str, purchase_token: str) -> dict[str, Any]:
    service = build_android_publisher()
    return (
        service.purchases()
        .subscriptionsv2()
        .get(packageName=package_name, token=purchase_token)
        .execute()
    )


def upsert_entitlement(
    db,
    firebase_uid: str,
    email: Optional[str],
    user_id: str,
    purchase_token: str,
    product_id: str,
    payload: dict[str, Any],
) -> SubscriptionEntitlement:
    """
    Vincula un purchase_token a UN SOLO usuario Firebase.

    Google Play restaura compras según la cuenta Google del dispositivo,
    no según el login interno de Firebase. Si el mismo token ya fue vinculado
    a otro firebase_uid, NO debe reasignarse automáticamente a la cuenta actual.
    Esto evita que una suscripción comprada por una cuenta se arrastre a otra.
    """
    plan = map_plan_from_product_id(product_id)
    status, is_active = map_status_from_google_payload(payload)
    expiry_date = extract_expiry_date(payload)
    latest_order_id = extract_order_id(payload)

    existing = (
        db.query(SubscriptionEntitlement)
        .filter(SubscriptionEntitlement.purchase_token == purchase_token)
        .first()
    )

    if existing is not None and existing.firebase_uid != firebase_uid:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "purchase_token_already_bound",
                "message": "Esta compra ya está vinculada a otra cuenta de PsicologIA.",
            },
        )

    if existing is None:
        existing = SubscriptionEntitlement(
            user_id=user_id,
            firebase_uid=firebase_uid,
            email=email,
            product_id=product_id,
            purchase_token=purchase_token,
            plan=plan,
            source="google_play",
            status=status,
            is_active=is_active,
            expiry_date=expiry_date,
            latest_order_id=latest_order_id,
            raw_payload=json.dumps(payload, ensure_ascii=False),
        )
        db.add(existing)
    else:
        existing.user_id = user_id
        existing.email = email
        existing.product_id = product_id
        existing.plan = plan
        existing.status = status
        existing.is_active = is_active
        existing.expiry_date = expiry_date
        existing.latest_order_id = latest_order_id
        existing.raw_payload = json.dumps(payload, ensure_ascii=False)
        existing.updated_at = now_utc()

    db.commit()
    db.refresh(existing)
    return existing


def get_current_entitlement(db, firebase_uid: str) -> Optional[SubscriptionEntitlement]:
    return (
        db.query(SubscriptionEntitlement)
        .filter(SubscriptionEntitlement.firebase_uid == firebase_uid)
        .order_by(SubscriptionEntitlement.is_active.desc(), SubscriptionEntitlement.updated_at.desc())
        .first()
    )


def save_chat_turn(db, user_id: str, role: str, content: str):
    db.add(ChatTurn(user_id=user_id, role=role, content=content))
    db.commit()


def save_memory(db, user_id: str, kind: str, content: str):
    db.add(MemoryItem(user_id=user_id, kind=kind, content=content))
    db.commit()


def save_journal(db, user_id: str, title: str, text: str, emotion: str, intensity: int):
    db.add(JournalEntry(user_id=user_id, title=title, text=text, emotion=emotion, intensity=intensity))
    db.commit()


def get_recent_chat(db, user_id: str, limit: int = 8):
    rows = db.query(ChatTurn).filter(ChatTurn.user_id == user_id).order_by(ChatTurn.created_at.desc()).limit(limit).all()
    rows.reverse()
    return rows


def get_recent_memories(db, user_id: str, limit: int = 5):
    return db.query(MemoryItem).filter(MemoryItem.user_id == user_id).order_by(MemoryItem.created_at.desc()).limit(limit).all()


def get_recent_journal(db, user_id: str, limit: int = 5):
    return db.query(JournalEntry).filter(JournalEntry.user_id == user_id).order_by(JournalEntry.created_at.desc()).limit(limit).all()


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
                {"role": "system", "content": "Eres una IA que analiza diarios emocionales y responde solo JSON válido en español."},
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
        "firebase_configured": bool(FIREBASE_SERVICE_ACCOUNT_PATH),
        "google_play_configured": bool(GOOGLE_PLAY_SERVICE_ACCOUNT_PATH),
        "package_name": ANDROID_PACKAGE_NAME,
    }


@app.get("/billing/my-plan")
def get_my_plan(authorization: Optional[str] = Header(default=None)):
    decoded = verify_firebase_token(authorization)
    firebase_uid = decoded["uid"]

    db = SessionLocal()
    try:
        row = get_current_entitlement(db, firebase_uid)
        if row is None:
            return {"plan": "free", "status": "inactive", "is_active": False, "expiry_date": None, "product_id": None}

        if row.expiry_date and row.expiry_date < now_utc():
            row.is_active = False
            row.status = "expired"
            db.commit()
            return {
                "plan": "free",
                "status": "expired",
                "is_active": False,
                "expiry_date": row.expiry_date.isoformat(),
                "product_id": row.product_id,
            }

        return {
            "plan": row.plan if row.is_active else "free",
            "status": row.status,
            "is_active": row.is_active,
            "expiry_date": row.expiry_date.isoformat() if row.expiry_date else None,
            "product_id": row.product_id,
        }
    finally:
        db.close()


@app.post("/billing/verify")
def verify_subscription(req: VerifySubscriptionRequest, authorization: Optional[str] = Header(default=None)):
    decoded = verify_firebase_token(authorization)
    firebase_uid = decoded["uid"]
    email = decoded.get("email")
    package_name = (req.package_name or ANDROID_PACKAGE_NAME).strip()

    if not req.purchase_token.strip():
        raise HTTPException(status_code=400, detail="purchase_token obligatorio")
    if not req.product_id.strip():
        raise HTTPException(status_code=400, detail="product_id obligatorio")

    db = SessionLocal()
    try:
        payload = google_verify_subscription(package_name=package_name, purchase_token=req.purchase_token.strip())
        row = upsert_entitlement(
            db=db,
            firebase_uid=firebase_uid,
            email=email,
            user_id=f"firebase::{firebase_uid}",
            purchase_token=req.purchase_token.strip(),
            product_id=req.product_id.strip(),
            payload=payload,
        )

        return {
            "ok": True,
            "plan": row.plan if row.is_active else "free",
            "status": row.status,
            "is_active": row.is_active,
            "expiry_date": row.expiry_date.isoformat() if row.expiry_date else None,
            "product_id": row.product_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo verificar la suscripción: {str(e)}")
    finally:
        db.close()


@app.post("/billing/restore")
def restore_subscription(req: RestoreSubscriptionRequest, authorization: Optional[str] = Header(default=None)):
    decoded = verify_firebase_token(authorization)
    firebase_uid = decoded["uid"]
    email = decoded.get("email")
    package_name = (req.package_name or ANDROID_PACKAGE_NAME).strip()

    if not req.purchase_tokens:
        raise HTTPException(status_code=400, detail="Debes enviar al menos un purchase_token")

    db = SessionLocal()
    restored_rows = []
    errors = []

    try:
        for token in req.purchase_tokens:
            try:
                payload = google_verify_subscription(package_name=package_name, purchase_token=token.strip())
                product_id = req.product_id_hint or ""
                line_items = payload.get("lineItems") or []
                if not product_id and line_items:
                    product_id = (line_items[0].get("productId") or "").strip()

                if not product_id:
                    errors.append({"purchase_token": token, "error": "No se pudo determinar product_id"})
                    continue

                row = upsert_entitlement(
                    db=db,
                    firebase_uid=firebase_uid,
                    email=email,
                    user_id=f"firebase::{firebase_uid}",
                    purchase_token=token.strip(),
                    product_id=product_id,
                    payload=payload,
                )
                restored_rows.append(row)
            except HTTPException as e:
                errors.append({"purchase_token": token, "error": e.detail})
            except Exception as e:
                errors.append({"purchase_token": token, "error": str(e)})

        current = get_current_entitlement(db, firebase_uid)

        return {
            "ok": True,
            "restored_count": len(restored_rows),
            "errors": errors,
            "plan": current.plan if current and current.is_active else "free",
            "status": current.status if current else "inactive",
            "is_active": current.is_active if current else False,
            "expiry_date": current.expiry_date.isoformat() if current and current.expiry_date else None,
            "product_id": current.product_id if current else None,
        }
    finally:
        db.close()


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
