from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class ContentBrief(BaseModel):
  model_config = ConfigDict(str_strip_whitespace=True)
  
  company_id: str = Field(min_length=1, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",)
  content_type: Literal["brochure"] = "brochure"
  instructions: str = Field(min_length=1)
  target_audience: str = ""
  tone_override: str | None = None