import logging
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from schemas import ExtraerEntidadesTecnicas

logger = logging.getLogger(__name__)

TEMPERATURE = 0
OPENAI_MODEL_NAME = "gpt-4o-mini"
ANTHROPIC_MODEL_NAME = "claude-3-5-haiku-latest"
STOP_AFTER_ATTEMPT = 3
TRUNCATED_REASONS = frozenset({"length", "max_tokens"})

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un analista de sistemas. Extrae entidades técnicas del texto. "
            "Identifica tecnologías mencionadas, el nivel_de_criticidad "
            "(baja, media o alta) y un resumen_tecnico breve. "
            "No inventes tecnologías que no aparezcan en el texto.",
        ),
        ("human", "{texto}"),
    ]
)


class IncompleteOutputError(OutputParserException):
    """La generación se cortó (finish_reason length o max_tokens)."""


def chat_model() -> ChatOpenAI | ChatAnthropic:
    """Modelo según LLM_PROVIDER. El retry de JSON lo pone LangChain, no el SDK."""
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower() or "openai"
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise RuntimeError("Falta OPENAI_API_KEY para el proveedor openai.")
        return ChatOpenAI(
            model_name=OPENAI_MODEL_NAME,
            temperature=TEMPERATURE,
            max_retries=0,
        )
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise RuntimeError("Falta ANTHROPIC_API_KEY para el proveedor anthropic.")
        return ChatAnthropic(
            model_name=ANTHROPIC_MODEL_NAME,
            temperature=TEMPERATURE,
            max_retries=0,
        )
    raise RuntimeError(f"LLM_PROVIDER no reconocido: {provider}")


def _stop_reason(raw: object) -> str | None:
    metadata = getattr(raw, "response_metadata", None) or {}
    additional = getattr(raw, "additional_kwargs", None) or {}
    sources = (
        metadata if isinstance(metadata, dict) else {},
        additional if isinstance(additional, dict) else {},
    )
    for source in sources:
        for key in ("finish_reason", "stop_reason"):
            value = source.get(key)
            if value:
                return str(value).lower()
    return None


def ensure_complete(payload: dict) -> ExtraerEntidadesTecnicas:
    """Valida parsed + finish_reason de with_structured_output(include_raw=True)."""
    parsed = payload.get("parsed")
    parsing_error = payload.get("parsing_error")
    raw = payload.get("raw")

    if parsing_error is not None or parsed is None:
        logger.warning("validación fallida: parsed ausente o parsing_error=%s", parsing_error)
        raise OutputParserException(
            str(parsing_error) if parsing_error is not None else "salida estructurada vacía"
        )

    reason = _stop_reason(raw)
    if reason in TRUNCATED_REASONS:
        logger.warning("objeto incompleto: stop/finish_reason=%s", reason)
        raise IncompleteOutputError(
            f"generación truncada por tokens (finish_reason={reason})"
        )

    if not isinstance(parsed, ExtraerEntidadesTecnicas):
        logger.warning("parsed no es ExtraerEntidadesTecnicas: %s", type(parsed))
        raise OutputParserException("la salida parseada no cumple ExtraerEntidadesTecnicas")

    return parsed


def build_chain():
    model = chat_model()
    structured = model.with_structured_output(
        ExtraerEntidadesTecnicas,
        include_raw=True,
    )
    core = prompt | structured | RunnableLambda(ensure_complete)
    return core.with_retry(
        retry_if_exception_type=(
            OutputParserException,
            ValidationError,
            IncompleteOutputError,
        ),
        wait_exponential_jitter=True,
        stop_after_attempt=STOP_AFTER_ATTEMPT,
    )


async def process_text(text: str) -> ExtraerEntidadesTecnicas:
    load_dotenv()
    logger.info("inicio de process_text")
    result = await build_chain().ainvoke({"texto": text})
    logger.info("éxito: %s", result.model_dump_json())
    return result
