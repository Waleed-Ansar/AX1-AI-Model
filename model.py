from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional, List, Union


class API_Response_Model(BaseModel):
    success: bool = False
    message: Optional[str] = None
    error: Optional[str] = None
    data: Optional[str] = None
    user_id: Optional[int] = None

class Feedback_Response_Model(BaseModel):
    success: bool = False
    error: Optional[str] = None
    data: Optional[str] = None
    user_id: Optional[int] = None

class Request_Model(BaseModel):
    user_id: str
    url: str
    categories: list

class Feedback_Model(BaseModel):
    user_id: str

@dataclass
class dtype():
    Organization: str
    Amount: float
    Suggestion: str
    