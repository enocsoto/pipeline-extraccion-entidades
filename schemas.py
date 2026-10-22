from enum import Enum

from pydantic import BaseModel, field_validator


class NivelDeCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraerEntidadesTecnicas(BaseModel):
    """Contrato de extracción técnica para with_structured_output."""

    tecnologias: list[str]
    nivel_de_criticidad: NivelDeCriticidad
    resumen_tecnico: str

    @field_validator("tecnologias")
    @classmethod
    def tecnologias_presentes(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("tecnologias no puede estar vacía")
        cleaned: list[str] = []
        for item in value:
            stripped = item.strip()
            if not stripped:
                raise ValueError("cada tecnología debe tener texto")
            cleaned.append(stripped)
        return cleaned

    @field_validator("resumen_tecnico")
    @classmethod
    def resumen_presente(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("resumen_tecnico no puede estar en blanco")
        return stripped
