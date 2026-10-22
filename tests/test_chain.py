"""Contrato Pydantic, cadena LCEL y guardia de finish_reason. Sin llamadas a la API."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

import chain
from schemas import ExtraerEntidadesTecnicas, NivelDeCriticidad

EXAMPLE_JSON = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": (
        "API con caché en Redis y persistencia en PostgreSQL; "
        "cuello de botella en conexiones concurrentes."
    ),
}


class _FakeChain:
    def __init__(self, result=None) -> None:
        self.result = result
        self.payload = None

    async def ainvoke(self, payload):
        self.payload = payload
        return self.result


def _example_model() -> ExtraerEntidadesTecnicas:
    return ExtraerEntidadesTecnicas.model_validate(EXAMPLE_JSON)


class TestExtraerEntidadesTecnicas(unittest.TestCase):
    def test_accepts_example_json(self) -> None:
        extracted = ExtraerEntidadesTecnicas.model_validate(EXAMPLE_JSON)
        self.assertEqual(extracted.tecnologias, ["FastAPI", "Redis", "PostgreSQL"])
        self.assertEqual(extracted.nivel_de_criticidad, NivelDeCriticidad.ALTA)
        self.assertIn("cuello de botella", extracted.resumen_tecnico)

    def test_rejects_empty_tecnologias_blank_item_blank_resumen_and_invalid_criticidad(
        self,
    ) -> None:
        with self.assertRaises(ValidationError):
            ExtraerEntidadesTecnicas(
                tecnologias=[],
                nivel_de_criticidad=NivelDeCriticidad.BAJA,
                resumen_tecnico="API estable.",
            )
        with self.assertRaises(ValidationError):
            ExtraerEntidadesTecnicas(
                tecnologias=["FastAPI", "  "],
                nivel_de_criticidad=NivelDeCriticidad.MEDIA,
                resumen_tecnico="API estable.",
            )
        with self.assertRaises(ValidationError):
            ExtraerEntidadesTecnicas(
                tecnologias=["FastAPI"],
                nivel_de_criticidad=NivelDeCriticidad.BAJA,
                resumen_tecnico="   ",
            )
        with self.assertRaises(ValidationError):
            ExtraerEntidadesTecnicas.model_validate(
                {
                    "tecnologias": ["FastAPI"],
                    "nivel_de_criticidad": "crítica",
                    "resumen_tecnico": "API estable.",
                }
            )

    def test_json_schema_omits_constraints_openai_strict_mode_rejects(self) -> None:
        dumped = json.dumps(ExtraerEntidadesTecnicas.model_json_schema())
        for keyword in (
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
            "minItems",
        ):
            self.assertNotIn(f'"{keyword}"', dumped)


class TestBuildChain(unittest.TestCase):
    def test_prompt_is_first_retry_is_three_and_source_uses_lcel(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            built = chain.build_chain()
        inner = getattr(built, "bound", built)
        self.assertIs(inner.first, chain.prompt)
        self.assertIn("texto", chain.prompt.input_variables)
        self.assertEqual(built.max_attempt_number, 3)

        source = Path(chain.__file__).read_text(encoding="utf-8")
        self.assertIn("prompt |", source)
        self.assertIn(".with_structured_output(", source)
        self.assertIn(".with_retry(", source)


class TestProcessText(unittest.IsolatedAsyncioTestCase):
    async def test_ainvoke_sends_texto_and_returns_schema(self) -> None:
        expected = _example_model()
        fake = _FakeChain(result=expected)
        with patch.object(chain, "build_chain", return_value=fake):
            result = await chain.process_text("log de FastAPI y Redis")
        self.assertEqual(result, expected)
        self.assertEqual(fake.payload, {"texto": "log de FastAPI y Redis"})


class TestEnsureComplete(unittest.TestCase):
    def test_returns_parsed_model_when_complete(self) -> None:
        parsed = _example_model()
        raw = SimpleNamespace(response_metadata={"finish_reason": "stop"})
        result = chain.ensure_complete(
            {"raw": raw, "parsed": parsed, "parsing_error": None}
        )
        self.assertEqual(result, parsed)

    def test_raises_when_parsed_is_none(self) -> None:
        raw = SimpleNamespace(response_metadata={"finish_reason": "stop"})
        with self.assertRaises(chain.OutputParserException):
            chain.ensure_complete(
                {"raw": raw, "parsed": None, "parsing_error": None}
            )

    def test_raises_when_finish_reason_is_length(self) -> None:
        parsed = _example_model()
        raw = SimpleNamespace(response_metadata={"finish_reason": "length"})
        with self.assertRaises(chain.IncompleteOutputError):
            chain.ensure_complete(
                {"raw": raw, "parsed": parsed, "parsing_error": None}
            )

    def test_raises_when_stop_reason_is_max_tokens(self) -> None:
        parsed = _example_model()
        raw = SimpleNamespace(response_metadata={"stop_reason": "max_tokens"})
        with self.assertRaises(chain.IncompleteOutputError):
            chain.ensure_complete(
                {"raw": raw, "parsed": parsed, "parsing_error": None}
            )


if __name__ == "__main__":
    unittest.main()
