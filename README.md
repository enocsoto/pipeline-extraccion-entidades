# Pipeline de extracción de entidades técnicas

Cadena LCEL que recibe un párrafo (log o descripción de arquitectura) y devuelve un objeto Pydantic validado: tecnologías, criticidad y resumen técnico. Si el JSON sale mal formado, incompleto o cortado por tokens, reintenta la generación.

## Cómo instalar

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env
```

Completa en `.env` la clave del proveedor (`OPENAI_API_KEY` o `ANTHROPIC_API_KEY`) y, si usas Anthropic, `LLM_PROVIDER=anthropic`.

## Cómo correr el ejemplo

```bash
.venv/bin/python main.py
```

`main.py` pasa un texto de ejemplo (FastAPI, Redis, PostgreSQL y un cuello de botella) a `process_text` y imprime el JSON.

Salida esperada (forma del objeto validado):

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

## Tests

No llaman a la API.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Variables de entorno

| Variable | Uso |
| --- | --- |
| `LLM_PROVIDER` | `openai` (por defecto) o `anthropic` |
| `OPENAI_API_KEY` | Obligatoria si el proveedor es `openai` |
| `ANTHROPIC_API_KEY` | Obligatoria si el proveedor es `anthropic` |

No pongas claves en el código ni en el README. `load_dotenv()` las lee en `process_text`.

## Cómo está armada la cadena

- **LCEL:** el operador `|` compone runnables. Aquí: `prompt | model.with_structured_output(ExtraerEntidadesTecnicas) | ensure_complete`.
- **ChatPromptTemplate:** mensajes `system` + `human` con la variable `{texto}`. No se arman prompts con f-strings.
- **with_structured_output:** pide al proveedor un JSON que cumpla `ExtraerEntidadesTecnicas`. Se usa `include_raw=True` para inspeccionar `finish_reason` / `stop_reason`.
- **ensure_complete:** si `parsed` es `None`, hay `parsing_error`, o el corte fue por `length` / `max_tokens`, lanza un error reintentable (JSON incompleto).
- **with_retry:** envuelve la cadena completa (`stop_after_attempt=3`, `wait_exponential_jitter=True`). El SDK del chat va con `max_retries=0` para no duplicar reintentos.
- **process_text:** función `async` que hace `.ainvoke({"texto": ...})` y registra inicio, éxito y fallos de validación.

Archivos: `schemas.py` (contrato), `chain.py` (cadena y `process_text`), `main.py` (script de prueba).
