from typing import Annotated, Literal, TypedDict
import operator

AgentName = Literal["revenue_agent", "customer_agent", "product_agent"]

class BusinessState(TypedDict, total=False):
    question: str
    plan: list[AgentName]
    findings: Annotated[list[dict], operator.add]
    validation_result: str
    validation_reason: str
    research_round: int
    research_note: str
    final_report: str

class SpecialistTask(TypedDict, total=False):
    question: str
    task: str
    findings: Annotated[list[dict], operator.add]


