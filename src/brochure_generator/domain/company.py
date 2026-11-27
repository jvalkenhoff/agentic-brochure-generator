from pydantic import BaseModel, ConfigDict, Field, HttpUrl

class CompanyProfile(BaseModel):
  model_config = ConfigDict(str_strip_whitespace=True)
  
  id: str = Field(min_length=1, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
  name: str = Field(min_length=1)
  website_url: HttpUrl
  
  description: str = ""
  brand_voice: str = ""