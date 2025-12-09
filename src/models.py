from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Profile(BaseModel):
    name: str
    headline: Optional[str] = ""
    bio: Optional[str] = ""
    location: Optional[str] = ""


class WorkExperience(BaseModel):
    title: str
    companyname: str
    description: Optional[str] = ""


class Education(BaseModel):
    school: str
    degree: Optional[str] = ""
    fieldofstudy: Optional[str] = ""


class Professional(BaseModel):
    workexperience: List[WorkExperience] = []
    education: List[Education] = []


class Actor(BaseModel):
    profile: Profile
    professional: Professional

    @property
    def unique_id(self) -> str:
        base = self.profile.name.lower().replace(" ", "_").strip()
        return "".join(c for c in base if c.isalnum() or c == "_")


class VectorRecord(BaseModel):
    id: str
    values: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
