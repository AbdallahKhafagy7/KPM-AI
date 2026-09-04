from pydantic import BaseModel

class lesson(BaseModel):
    id: int
    title:str 
    project_name: str
    country: str
    vendor: str
    client: str
    industry:str
    value_proposition: str
    description: str
    content: list[str]
    keywords: list[str]
    category: str
    status: str