from pydantic import BaseModel


class PatientInstances(BaseModel):
    name: str
    label: int
    features: list[list[float]]
