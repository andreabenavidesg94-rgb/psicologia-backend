import hashlib
import json
import logging
import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

# Logger de billing — todos los errores de Google Play aparecen en Render Logs
_log = logging.getLogger("psicologia.billing")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

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


def as_utc_aware(value: Optional[datetime]) -> Optional[datetime]:
    """
    Normaliza fechas para evitar el error:
    TypeError: can't compare offset-naive and offset-aware datetimes.

    Algunas bases de datos/devuelven DateTime sin tzinfo aunque Google Play
    venga con +00:00. Para comparar siempre usamos UTC aware.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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
    focus_area: str | None = "general"
    companion_style: str | None = "suave"   # suave | directa | motivadora | reflexiva
    remembered_topic: str | None = ""
    user_id: str | None = None
    emotional_context: str | None = ""


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
    platform: Optional[str] = "android"


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
        # En Google Play, cancelada no siempre significa vencida:
        # el usuario puede haber cancelado la renovación, pero conserva acceso hasta expiryTime.
        expiry_date = extract_expiry_date(payload)
        if expiry_date and expiry_date > now_utc():
            return "active", True
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
        latest = item.get("latestSuccessfulOrderId") or item.get("latest_successful_order_id")
        if latest:
            return str(latest)
    return None


def extract_product_id_from_payload(payload: dict[str, Any], hint: str = "") -> str:
    """
    Extrae el product_id del payload de subscriptionsv2 de Google Play.

    La API devuelve lineItems[].productId.
    Si hint está presente y es un product_id conocido, lo usa directamente
    (evita ambigüedad cuando hay múltiples lineItems).
    """
    if hint and hint.strip():
        return hint.strip()

    line_items = payload.get("lineItems") or []
    for item in line_items:
        pid = (item.get("productId") or "").strip()
        if pid:
            _log.info(f"product_id extraído de lineItems: {pid}")
            return pid

    # Fallback: intentar leer desde el campo de nivel raíz (versiones antiguas de la API)
    pid = (payload.get("productId") or "").strip()
    if pid:
        _log.info(f"product_id extraído de raíz del payload: {pid}")
        return pid

    _log.warning(
        f"No se pudo extraer product_id del payload. "
        f"subscriptionState={payload.get('subscriptionState')} "
        f"lineItems_count={len(line_items)}"
    )
    return ""


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
    expiry_date = as_utc_aware(extract_expiry_date(payload))
    latest_order_id = extract_order_id(payload)

    if expiry_date and expiry_date < now_utc():
        status = "expired"
        is_active = False

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
    row = (
        db.query(SubscriptionEntitlement)
        .filter(SubscriptionEntitlement.firebase_uid == firebase_uid)
        .order_by(SubscriptionEntitlement.is_active.desc(), SubscriptionEntitlement.updated_at.desc())
        .first()
    )

    row_expiry = as_utc_aware(row.expiry_date) if row and row.expiry_date else None
    if row and row_expiry and row_expiry < now_utc():
        row.is_active = False
        row.status = "expired"
        row.updated_at = now_utc()
        db.commit()
        db.refresh(row)

    return row


def refresh_entitlement_with_google(db, row: SubscriptionEntitlement) -> SubscriptionEntitlement:
    """
    Revalida automáticamente el purchase_token guardado.
    Esto evita que el usuario tenga que pulsar Restaurar compras cada vez que abre la app.
    """
    token_preview = (row.purchase_token or "")[:8] + "…"
    try:
        payload = google_verify_subscription(
            package_name=ANDROID_PACKAGE_NAME,
            purchase_token=row.purchase_token,
        )
        product_id = extract_product_id_from_payload(payload, hint=row.product_id or "")
        refreshed = upsert_entitlement(
            db=db,
            firebase_uid=row.firebase_uid,
            email=row.email,
            user_id=row.user_id,
            purchase_token=row.purchase_token,
            product_id=product_id or row.product_id,
            payload=payload,
        )
        _log.info(
            f"my-plan auto refresh OK | uid={row.firebase_uid[:8]}… | token={token_preview} | "
            f"plan={refreshed.plan} | active={refreshed.is_active} | status={refreshed.status}"
        )
        return refreshed
    except Exception as exc:
        _log.error(
            f"my-plan auto refresh failed | uid={row.firebase_uid[:8]}… | token={token_preview} | "
            f"error={type(exc).__name__}: {exc}"
        )
        return row


def entitlement_response(row: Optional[SubscriptionEntitlement]) -> dict[str, Any]:
    if row is None:
        return {"plan": "free", "status": "inactive", "is_active": False, "expiry_date": None, "product_id": None}
    return {
        "plan": row.plan if row.is_active else "free",
        "status": row.status,
        "is_active": row.is_active,
        "expiry_date": row.expiry_date.isoformat() if row.expiry_date else None,
        "product_id": row.product_id,
    }


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
    nombre = (req.user_name or "").strip() or "Tú"

    if any(x in texto for x in ["ansiedad", "nervios", "ansiosa", "ansioso"]):
        return (
            f"Lo que describes suena a que tu sistema de alerta está muy activo ahora mismo. "
            f"Vamos a bajar un poco el ritmo antes de analizar nada. "
            f"¿Qué fue lo primero que lo disparó hoy, si pudieras identificar un momento concreto?"
        )
    if any(x in texto for x in ["pareja", "relación", "me ignoró", "no respondió", "me dejó"]):
        return (
            "Cuando hay silencio o distancia de alguien que importa, la mente empieza a construir historias rápido. "
            "Eso duele de forma muy real. "
            "¿Qué fue lo primero que pensaste cuando pasó eso?"
        )
    if any(x in texto for x in ["agot", "cans", "burnout", "no puedo más"]):
        return (
            "Eso suena a más que cansancio físico: es ese punto en el que hasta las cosas pequeñas empiezan a pesar demasiado. "
            "Antes de pedirte que hagas algo, cuéntame un poco más: "
            "¿este agotamiento viene más de personas, de trabajo o de sentir que no paras nunca?"
        )
    if any(x in texto for x in ["sola", "solo", "nadie me entiende", "incomprendida"]):
        return (
            "Sentirse sola aunque haya gente alrededor es uno de los cansancios más silenciosos que existen. "
            "¿Hay algo concreto que pasó recientemente que lo intensificó, "
            "o es más una sensación que viene de hace tiempo?"
        )
    if any(x in texto for x in ["triste", "tristeza", "vacío", "vacía", "llorar", "llor"]):
        return (
            "La tristeza a veces no pide explicación, simplemente aparece y pesa. "
            "No voy a pedirte que la expliques si no puedes. "
            "¿Quieres contarme qué pasó, o prefieres que empecemos por lo que sientes en el cuerpo ahora mismo?"
        )
    if any(x in texto for x in ["dormir", "insomnio", "no puedo dormir", "desvelada"]):
        return (
            "Qué frustrante es cuando el cuerpo está cansado pero la mente no baja el volumen. "
            "No voy a decirte simplemente que te relajes, porque sé que no funciona así. "
            "¿Quieres que hagamos una respiración breve juntas, o prefieres vaciar primero lo que tienes en la cabeza?"
        )
    if any(x in texto for x in ["fatal", "horrible", "mal", "pésimo", "todo está mal"]):
        return (
            "Uf, sentirse así sin poder ponerle nombre exacto agota todavía más. "
            "No tenemos que resolver nada de golpe. "
            "¿Se siente más como tristeza, ansiedad, cansancio o algo más como estar saturada de todo?"
        )
    if any(x in texto for x in ["hola", "buenas", "buenos días", "buenas tardes"]):
        return (
            f"Hola, me alegra que estés aquí. "
            f"¿Cómo estás hoy, de verdad? No hace falta que todo esté bien ni mal, solo cuéntame."
        )
    return (
        f"Te escucho. "
        f"¿Quieres empezar por contarme qué pasó, cómo te sientes ahora mismo, "
        f"o prefieres que te ayude a ordenar lo que tienes en la cabeza?"
    )


_STYLE_INSTRUCTIONS = {
    "suave": (
        "Tu tono es suave, cálido, sin prisa. Validas antes de proponer nada. "
        "Usas frases cortas y naturales. No das listas de consejos."
    ),
    "directa": (
        "Tu tono es claro y directo, sin rodeos, pero con calidez. "
        "Vas al punto importante rápido. Preguntas concretas, no vagas."
    ),
    "motivadora": (
        "Tu tono es animador, activo, con energía positiva real (no forzada). "
        "Destacas los recursos que ya tiene la persona. Invitas a la acción pequeña."
    ),
    "reflexiva": (
        "Tu tono es pausado, contemplativo. Invitas a mirar hacia adentro. "
        "Haces preguntas que abren espacio, no que buscan solución inmediata."
    ),
}

_FOCUS_CONTEXT = {
    "ansiedad": "La persona está trabajando su ansiedad. Cuando aparezca activación, ayúdale a bajar el ritmo primero.",
    "estres": "La persona está manejando estrés. Ayúdale a identificar la fuente y a sentir que no está sola.",
    "dormir": "La persona tiene dificultades para dormir. Evita decirle simplemente que se relaje; ofrece algo concreto.",
    "desahogarme": "La persona necesita ser escuchada, no soluciones. Escucha primero, valida segundo, pregunta después.",
    "diario": "La persona quiere explorar sus emociones mediante escritura. Ayúdale a articular lo que siente.",
    "autoconocimiento": "La persona quiere conocerse mejor. Ayúdale a notar patrones y a observarse sin juzgarse.",
    "general": "Adapta tu respuesta al contenido emocional del mensaje.",
}


def build_system_prompt(req: ChatRequest, memory_lines: list[str], journal_lines: list[str]) -> str:
    memories = "\n".join(f"- {m}" for m in memory_lines) if memory_lines else "- Sin memoria previa relevante."
    journals = "\n".join(f"- {j}" for j in journal_lines) if journal_lines else "- Sin registros recientes."

    style_key = (req.companion_style or "suave").lower()
    style_instr = _STYLE_INSTRUCTIONS.get(style_key, _STYLE_INSTRUCTIONS["suave"])

    focus_key = (req.focus_area or "general").lower()
    focus_instr = _FOCUS_CONTEXT.get(focus_key, _FOCUS_CONTEXT["general"])

    return f"""Eres {req.assistant_name}, una asistente emocional de la app PsicologIA.
Hablas en español. Eres humana, cálida y específica — nada genérico.

TONO: {style_instr}

CONTEXTO DE LA PERSONA:
- Nombre: {req.user_name}
- Área principal: {focus_key}
- Instrucción de contexto: {focus_instr}

MEMORIA EMOCIONAL:
{memories}

DIARIO RECIENTE:
{journals}

REGLAS DE RESPUESTA — sigue TODAS:
1. Refleja algo CONCRETO del mensaje del usuario. No respondas de forma genérica.
2. Valida emocionalmente sin exagerar: "Eso suena pesado" > "Claro que sí, entiendo perfectamente cómo te sientes".
3. Varía tus preguntas. Evita repetir siempre "¿cómo te sientes con eso?" o "¿quieres contarme más?".
4. Cuando el usuario parece bloqueado, ofrece opciones: "¿Quieres desahogarte, ordenar lo que pasó o calmarte primero?"
5. Responde con 3 a 5 frases. Ni más corto (frío) ni más largo (agotador).
6. No uses frases de plantilla como "Estoy aquí para ti", "Te escucho", "Entiendo lo que dices" de forma repetida.
7. No suenes como psicóloga clínica rígida. Suena como una amiga formada y empática.
8. No des diagnósticos. No uses lenguaje técnico sin explicarlo.
9. Mantén el hilo de la conversación reciente. No empieces desde cero.
10. Si el usuario habló de algo (pareja, trabajo, insomnio), continúa ESE hilo.
11. Habla como una persona cercana y emocionalmente inteligente, NO como terapeuta clínico.
12. La conversación debe sentirse natural, cálida y humana.
13. A veces valida primero SIN hacer preguntas inmediatas.
14. Usa pequeñas expresiones humanas naturales como:
   "Uf..."
   "Eso pesa mucho."
   "Te entiendo."
   "Tiene sentido que te sientas así."
   "No debió ser fácil."
   "Qué agotador."
   pero sin exagerar ni sonar artificial.
15. No uses tono corporativo ni frases robóticas de bienestar.
16. Prioriza conexión emocional antes que consejos.
17. Evita responder como chatbot asistente virtual.
18. No enumeres soluciones como lista salvo que la persona las pida.
19. Si el usuario vuelve varias veces sobre un tema, muestra continuidad emocional:
   "Siento que esto lleva tiempo pesándote."
   "Parece que esto te viene agotando desde hace días."
20. Las respuestas deben sentirse como conversación de confianza entre dos personas.
21. A veces una respuesta corta y cálida es mejor que una explicación larga.
22. No fuerces optimismo.
23. Si el usuario está vulnerable, habla con más suavidad y menos preguntas.
24. Nunca hagas sentir al usuario analizado psicológicamente.
25. La app acompaña, escucha y ayuda a ordenar emociones; no diagnostica.
26. Usa lenguaje moderno, natural y cercano, especialmente para usuarios jóvenes adultos.
27. Evita repetir constantemente:
   "¿Cómo te sientes?"
   "Estoy aquí para ti."
   "Entiendo cómo te sientes."
28. Mantén variedad emocional y naturalidad en cada respuesta.
29. Si el usuario comparte algo doloroso, primero acompaña emocionalmente antes de analizar.
30. Haz que el usuario sienta calma, compañía y seguridad emocional.

SEGURIDAD:
Si detectas riesgo grave, autolesión o suicidio:
- Prioriza seguridad antes que cualquier otra cosa.
- Recomienda ayuda inmediata (emergencias, persona de confianza, línea de crisis).
- Sé directa y cálida, no clínica ni alarmante.
- En España: Teléfono de la Esperanza 717 003 717. En Colombia: Línea 106.

EJEMPLOS DE BUEN TONO (úsalos como referencia de estilo, no como plantilla):
Usuario: "No sé qué me pasa últimamente."
Asistente: "A veces uno llega a un punto donde ya no sabe si está cansado, triste o simplemente saturado de todo. ¿Sientes que te viene pasando seguido?"

Usuario: "Me siento solo."
Asistente: "La soledad pesa más cuando llevas días guardándote todo. ¿Has podido hablar con alguien de cómo te sientes realmente?"

Usuario: "Estoy agotado."
Asistente: "Uf… cuando el cansancio ya es mental y emocional, hasta responder mensajes cuesta."

Usuario: "No quiero hacer nada."
Asistente: "Suena a que llevas demasiado tiempo sosteniendo cosas por dentro."

Usuario: "Tuve ansiedad otra vez."
Asistente: "¿Fue de esas veces donde sientes que tu cabeza va demasiado rápido y tu cuerpo no logra seguirle el ritmo?"
- Usuario: "Me siento fatal sin saber por qué." → "Uf, eso agota el doble: sentirse mal encima de no poder explicárselo. No tenemos que resolverlo ahora. ¿Se siente más como tristeza, ansiedad o simplemente vacío?"
- Usuario: "Mi pareja no respondió y me activé mucho." → "Entiendo por qué eso te activó. Muchas veces lo que duele no es el silencio, sino la historia que la mente empieza a construir. ¿Qué fue lo primero que pensaste?"
- Usuario: "No puedo dormir." → "Qué frustrante es cuando el cuerpo está cansado y la mente no baja el volumen. ¿Quieres que te guíe con una respiración breve o prefieres vaciar primero lo que tienes en la cabeza?"
""".strip()


def openai_chat_reply(
    req: ChatRequest,
    memory_lines: list[str],
    journal_lines: list[str],
    recent_chat_lines: list[str],
    emotional_context: str = "",
) -> str:
    if client is None:
        return fallback_chat_reply(req)

    system_prompt = build_system_prompt(req, memory_lines, journal_lines)
    transcript = "\n".join(recent_chat_lines[-8:]) if recent_chat_lines else "Sin conversación reciente."
    
    emotional_context_block = ""
    if emotional_context:
        emotional_context_block = f"""

    Contexto emocional reciente:
    {emotional_context}

    Usa este contexto de forma sutil, humana y no invasiva.
    No digas "según tus datos" ni "tu historial dice".
    Si ayuda, mantén continuidad emocional con frases como:
    "recuerdo que últimamente...", "podemos retomarlo con calma..." o
    "parece que este tema ha estado presente".
    """

    user_prompt = f"""
Conversación reciente:
{transcript}

{emotional_context_block}

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
            return entitlement_response(None)

        # Revalidación automática con Google Play usando el token ya guardado.
        # Si Google está temporalmente indisponible, devolvemos el último estado local.
        row = refresh_entitlement_with_google(db, row)
        return entitlement_response(row)
    finally:
        db.close()


@app.post("/billing/verify")
def verify_subscription(req: VerifySubscriptionRequest, authorization: Optional[str] = Header(default=None)):
    decoded = verify_firebase_token(authorization)
    firebase_uid = decoded["uid"]
    email = decoded.get("email")
    package_name = (req.package_name or ANDROID_PACKAGE_NAME).strip()
    platform = getattr(req, "platform", "android")

    if not req.purchase_token.strip():
        raise HTTPException(status_code=400, detail="purchase_token obligatorio")
    if not req.product_id.strip():
        raise HTTPException(status_code=400, detail="product_id obligatorio")

    db = SessionLocal()
    try:
        if platform == "ios":

            product_id = req.product_id.strip().lower()

            if "premium" in product_id:
                plan = "premium"
            elif "plus" in product_id:
                plan = "plus"
            else:
                plan = "free"
                
            apple_token_hash = "ios_" + hashlib.sha256(
                 req.purchase_token.strip().encode("utf-8")
            ).hexdigest()

            row = upsert_entitlement(
                db=db,
                firebase_uid=firebase_uid,
                email=email,
                user_id=f"firebase::{firebase_uid}",
                purchase_token=apple_token_hash,
                product_id=req.product_id.strip(),
                payload={
                    "platform": "ios",
                    "plan": plan,
                    "status": "active"
                }
            )
            
            row.plan = plan
            row.source = "ios"
            row.status = "active"
            row.is_active = True
            row.product_id = req.product_id.strip()

            db.commit()

            return {
                "ok": True,
                "platform": "ios",
                "plan": plan,
                "status": "active"
            }

        # ANDROID
        payload = google_verify_subscription(
            package_name=package_name,
            purchase_token=req.purchase_token.strip()
        )

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
    """
    Restaura suscripciones verificando cada purchase_token contra Google Play.

    Correcciones respecto a la versión anterior:
    - Todos los errores de Google Play se imprimen en logs (Render Logs).
    - El product_id se extrae con extract_product_id_from_payload (robusto).
    - El 409 token_already_bound se trata como warning, no como error silencioso.
    - Si Google Play devuelve error HTTP, se loguea el status y el motivo.
    - No se imprime el purchase_token completo en logs (solo los 8 primeros chars).
    - Si al menos un token se restaura correctamente, el plan resultante refleja
      el mejor entitlement activo del usuario.
    - Los tokens con error devuelven detail en errors[] para que Flutter pueda
      mostrar un mensaje informativo si lo necesita.
    """
    decoded = verify_firebase_token(authorization)
    firebase_uid = decoded["uid"]
    email = decoded.get("email")
    package_name = (req.package_name or ANDROID_PACKAGE_NAME).strip()

    if not req.purchase_tokens:
        raise HTTPException(status_code=400, detail="Debes enviar al menos un purchase_token")

    _log.info(
        f"restore_subscription start | uid={firebase_uid[:8]}… | "        f"tokens_count={len(req.purchase_tokens)} | package={package_name}"
    )

    db = SessionLocal()
    restored_rows = []
    errors = []

    try:
        for token in req.purchase_tokens:
            token_clean = token.strip()
            token_preview = token_clean[:8] + "…"  # Nunca imprimir el token completo

            # ── 1. Verificar con Google Play Developer API ────────────────────
            try:
                payload = google_verify_subscription(
                    package_name=package_name,
                    purchase_token=token_clean,
                )
            except Exception as gplay_exc:
                # Logueamos el error COMPLETO en Render Logs para diagnóstico
                _log.error(
                    f"Google Play verify failed | token={token_preview} | "                    f"error={type(gplay_exc).__name__}: {gplay_exc}"
                )
                errors.append({
                    "token_preview": token_preview,
                    "stage": "google_play_verify",
                    "error": f"{type(gplay_exc).__name__}: {str(gplay_exc)}",
                })
                continue  # Pasar al siguiente token

            # ── 2. Extraer product_id del payload (robusto) ───────────────────
            product_id = extract_product_id_from_payload(payload, hint=req.product_id_hint or "")

            if not product_id:
                _log.warning(
                    f"product_id vacío tras extracción | token={token_preview} | "                    f"state={payload.get('subscriptionState')}"
                )
                errors.append({
                    "token_preview": token_preview,
                    "stage": "product_id_extraction",
                    "error": "No se pudo determinar product_id desde el payload de Google Play",
                    "subscription_state": payload.get("subscriptionState", "unknown"),
                })
                continue

            # ── 3. Log del estado antes de guardar ────────────────────────────
            sub_state = payload.get("subscriptionState", "unknown")
            _log.info(
                f"Google Play OK | token={token_preview} | "                f"product_id={product_id} | state={sub_state}"
            )

            # ── 4. Guardar o actualizar en DB ─────────────────────────────────
            try:
                row = upsert_entitlement(
                    db=db,
                    firebase_uid=firebase_uid,
                    email=email,
                    user_id=f"firebase::{firebase_uid}",
                    purchase_token=token_clean,
                    product_id=product_id,
                    payload=payload,
                )
                restored_rows.append(row)
                _log.info(
                    f"entitlement upserted | token={token_preview} | "                    f"plan={row.plan} | is_active={row.is_active} | status={row.status}"
                )

            except HTTPException as h:
                # 409: token ya vinculado a otro firebase_uid — es un warning, no un crash
                detail = h.detail if isinstance(h.detail, str) else str(h.detail)
                _log.warning(
                    f"upsert HTTPException {h.status_code} | token={token_preview} | detail={detail}"
                )
                errors.append({
                    "token_preview": token_preview,
                    "stage": "upsert_entitlement",
                    "http_status": h.status_code,
                    "error": detail,
                })

            except Exception as db_exc:
                _log.error(
                    f"upsert DB error | token={token_preview} | "                    f"error={type(db_exc).__name__}: {db_exc}"
                )
                errors.append({
                    "token_preview": token_preview,
                    "stage": "upsert_entitlement",
                    "error": f"{type(db_exc).__name__}: {str(db_exc)}",
                })

        # ── 5. Leer el mejor entitlement activo del usuario ───────────────────
        current = get_current_entitlement(db, firebase_uid)
        final_plan = current.plan if (current and current.is_active) else "free"

        _log.info(
            f"restore_subscription done | uid={firebase_uid[:8]}… | "            f"restored={len(restored_rows)} | errors={len(errors)} | final_plan={final_plan}"
        )

        return {
            "ok": True,
            "restored_count": len(restored_rows),
            "errors": errors,
            "plan": final_plan,
            "status": current.status if current else "inactive",
            "is_active": current.is_active if current else False,
            "expiry_date": current.expiry_date.isoformat() if current and current.expiry_date else None,
            "product_id": current.product_id if current else None,
        }

    finally:
        db.close()


# ============================================================
# MEMORIA / RESUMEN SEMANAL / NOTIFICACIONES INTELIGENTES
# ============================================================

_EMOTION_RULES = [
    ("ansiedad", ["ansied", "ansios", "nerv", "pánico", "panico", "miedo", "preocup", "ataque"]),
    ("tristeza", ["triste", "vacío", "vacio", "llorar", "lloré", "llore", "solo", "sola", "desanim"]),
    ("agotamiento", ["agot", "cans", "satur", "estrés", "estres", "burnout", "colaps", "no puedo"]),
    ("sueño", ["dormir", "sueño", "sueno", "insom", "desvel", "noche"]),
    ("relaciones", ["pareja", "relación", "relacion", "amor", "familia", "hijo", "hija", "ruptura"]),
    ("autoexigencia", ["culpa", "fracaso", "exigir", "perfect", "debería", "deberia"]),
]

_EMOJI_BY_TOPIC = {
    "ansiedad": "😰",
    "tristeza": "😔",
    "agotamiento": "😵‍💫",
    "sueño": "🌙",
    "relaciones": "💚",
    "autoexigencia": "🧭",
    "calma": "🌿",
}


def _safe_lower(text: Optional[str]) -> str:
    return (text or "").lower()


def _detect_topic_from_text(text: str) -> str:
    lower = _safe_lower(text)
    scores: dict[str, int] = {}
    for topic, keys in _EMOTION_RULES:
        scores[topic] = sum(1 for k in keys if k in lower)
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "calma"


def _emotion_value_from_text(text: str) -> int:
    topic = _detect_topic_from_text(text)
    if topic in ["ansiedad", "agotamiento", "tristeza"]:
        return 2
    if topic in ["sueño", "autoexigencia", "relaciones"]:
        return 3
    return 4


def _compact_snippet(text: str, limit: int = 90) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "…"


def _topic_label(topic: str) -> str:
    return {
        "ansiedad": "ansiedad o nervios",
        "tristeza": "tristeza o soledad",
        "agotamiento": "agotamiento o saturación",
        "sueño": "descanso y sueño",
        "relaciones": "relaciones importantes",
        "autoexigencia": "autoexigencia",
        "calma": "tu bienestar emocional",
    }.get(topic, "tu bienestar emocional")


def _weekly_rows(db, firebase_uid: str, days: int = 7):
    since = now_utc().replace(tzinfo=None)  # compatibilidad con DateTime sin timezone en SQLAlchemy
    since = since.replace(hour=0, minute=0, second=0, microsecond=0)
    from datetime import timedelta
    since = since - timedelta(days=days - 1)

    chats = (
        db.query(ChatTurn)
        .filter(ChatTurn.user_id == firebase_uid, ChatTurn.role == "user", ChatTurn.created_at >= since)
        .order_by(ChatTurn.created_at.desc())
        .limit(40)
        .all()
    )
    journals = (
        db.query(JournalEntry)
        .filter(JournalEntry.user_id == firebase_uid, JournalEntry.created_at >= since)
        .order_by(JournalEntry.created_at.desc())
        .limit(40)
        .all()
    )
    return chats, journals


def _build_memory_payload(db, firebase_uid: str) -> dict[str, Any]:
    chats = (
        db.query(ChatTurn)
        .filter(ChatTurn.user_id == firebase_uid, ChatTurn.role == "user")
        .order_by(ChatTurn.created_at.desc())
        .limit(12)
        .all()
    )
    journals = (
        db.query(JournalEntry)
        .filter(JournalEntry.user_id == firebase_uid)
        .order_by(JournalEntry.created_at.desc())
        .limit(8)
        .all()
    )
    memories = (
        db.query(MemoryItem)
        .filter(MemoryItem.user_id == firebase_uid)
        .order_by(MemoryItem.created_at.desc())
        .limit(5)
        .all()
    )

    texts = [c.content for c in chats] + [j.text for j in journals] + [m.content for m in memories]
    if not texts:
        return {
            "ok": True,
            "has_memory": False,
            "topic": "calma",
            "topic_label": "tu bienestar emocional",
            "emoji": "🌿",
            "title": "PsicologIA",
            "body": "¿Quieres hacer un check-in breve y ver cómo estás hoy?",
            "payload": "daily_checkin",
        }

    topic_counts: dict[str, int] = {}
    for t in texts:
        topic = _detect_topic_from_text(t)
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    topic = max(topic_counts.items(), key=lambda x: x[1])[0]
    latest = texts[0]
    label = _topic_label(topic)

    if topic == "calma":
        body = "Tu rutina emocional sigue disponible. ¿Quieres revisar cómo estás hoy?"
    else:
        body = f"Recuerdo que últimamente apareció {label}. ¿Quieres revisar cómo sigues hoy?"

    return {
        "ok": True,
        "has_memory": True,
        "topic": topic,
        "topic_label": label,
        "emoji": _EMOJI_BY_TOPIC.get(topic, "🌿"),
        "title": "PsicologIA",
        "body": body,
        "latest_snippet": _compact_snippet(latest),
        "payload": "memory_followup",
    }


def _build_weekly_summary_payload(db, firebase_uid: str) -> dict[str, Any]:
    chats, journals = _weekly_rows(db, firebase_uid, days=7)
    texts = [c.content for c in chats] + [j.text for j in journals]
    days_with_activity = set()
    for row in list(chats) + list(journals):
        if row.created_at:
            days_with_activity.add(row.created_at.date().isoformat())

    if not texts:
        return {
            "ok": True,
            "has_data": False,
            "title": "Tu resumen semanal se está preparando",
            "body": "Cuando registres emociones o escribas en el diario, aquí aparecerán tus patrones con calma.",
            "main_topic": "calma",
            "main_topic_label": "tu bienestar emocional",
            "emoji": "🌿",
            "activity_days": 0,
            "journal_entries": 0,
            "chat_messages": 0,
            "suggestion": "Empieza con un check-in de 1 minuto.",
        }

    topic_counts: dict[str, int] = {}
    values: list[int] = []
    for t in texts:
        topic = _detect_topic_from_text(t)
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        values.append(_emotion_value_from_text(t))
    main_topic = max(topic_counts.items(), key=lambda x: x[1])[0]
    avg = sum(values) / max(len(values), 1)
    label = _topic_label(main_topic)

    if avg <= 2.2:
        tone = "Esta semana pidió más cuidado y menos exigencia."
        suggestion = "Haz una pausa breve, escribe dos líneas y retoma una conversación suave."
    elif avg <= 3.2:
        tone = "Hubo señales mixtas: algo pesó, pero también hubo continuidad."
        suggestion = "Elige una acción pequeña diaria: respirar, escribir o hablar 2 minutos."
    else:
        tone = "Tu semana se ve más estable y con mejor continuidad emocional."
        suggestion = "Mantén el hábito; lo importante es volver sin exigirte perfección."

    return {
        "ok": True,
        "has_data": True,
        "title": "Tu semana emocional en calma",
        "body": f"{tone} El patrón más visible fue {label}.",
        "main_topic": main_topic,
        "main_topic_label": label,
        "emoji": _EMOJI_BY_TOPIC.get(main_topic, "🌿"),
        "activity_days": len(days_with_activity),
        "journal_entries": len(journals),
        "chat_messages": len(chats),
        "suggestion": suggestion,
        "notification_body": f"Tu resumen semanal está listo. Esta semana destacó {label}. ¿Quieres verlo con calma?",
    }


@app.get("/memory/notification-context")
def memory_notification_context(authorization: Optional[str] = Header(default=None)):
    decoded = verify_firebase_token(authorization)
    firebase_uid = decoded["uid"]
    db = SessionLocal()
    try:
        return _build_memory_payload(db, firebase_uid)
    finally:
        db.close()


@app.get("/memory/weekly-summary")
def memory_weekly_summary(authorization: Optional[str] = Header(default=None)):
    decoded = verify_firebase_token(authorization)
    firebase_uid = decoded["uid"]
    db = SessionLocal()
    try:
        return _build_weekly_summary_payload(db, firebase_uid)
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
        emotional_context = (req.emotional_context or "").strip()

        respuesta = openai_chat_reply(
            req,
             memory_lines,
             journal_lines,
             recent_chat_lines,
             emotional_context
        )
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
