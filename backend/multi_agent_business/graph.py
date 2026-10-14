from langgraph.graph import StateGraph, START, END
from state import BusinessState
from supervisor import supervisor_plan, dispatch_to_specialists
from specialists import revenue_agent, customer_agent, product_agent
from validation import validation_agent, route_after_validation
from research import targeted_research
from reporting import generate_report

def build_graph():
    builder = StateGraph(BusinessState)

    builder.add_node("supervisor", supervisor_plan)
    builder.add_node("revenue_agent", revenue_agent)
    builder.add_node("customer_agent", customer_agent)
    builder.add_node("product_agent", product_agent)
    builder.add_node("validation_agent", validation_agent)
    builder.add_node("targeted_research", targeted_research)
    builder.add_node("generate_report", generate_report)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", dispatch_to_specialists)
    builder.add_edge("revenue_agent", "validation_agent")
    builder.add_edge("customer_agent", "validation_agent")
    builder.add_edge("product_agent", "validation_agent")

    builder.add_conditional_edges(
        "validation_agent",
        route_after_validation,
        {
            "report": "generate_report",
            "report_with_limitations": "generate_report",
            "targeted_research": "targeted_research",
        },
    )
    builder.add_edge("targeted_research", "validation_agent")
    builder.add_edge("generate_report", END)

    return builder.compile()
