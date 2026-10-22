import asyncio
import logging

from chain import process_text

logging.basicConfig(level=logging.INFO)

SAMPLE_TEXT = (
    "El servicio HTTP está en FastAPI, con caché en Redis y persistencia en PostgreSQL. "
    "El log muestra timeouts y un cuello de botella en las conexiones concurrentes a la base."
)


if __name__ == "__main__":
    resultado = asyncio.run(process_text(SAMPLE_TEXT))
    print(resultado.model_dump_json(indent=2))
