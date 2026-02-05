from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class IssuePayload(BaseModel):
    url: Optional[str] = None
    issueCode: Optional[str] = None
    title: Optional[str] = None
    reportedOn: Optional[str] = None
    severity: Optional[str] = None
    reproductionSteps: Optional[List[str]] = None
    whatHappened: Optional[str] = None
    whatShouldHaveHappened: Optional[str] = None
    workaround: Optional[str] = None

class MatchResponse(BaseModel):
    matches: List[Dict[str, Any]]
