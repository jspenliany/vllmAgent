from cstate.PersonState import PersonState
from langchain_core.messages import HumanMessage


def retrieve_memory_node(state: PersonState):
    # Real logic: db.similarity_search(state["messages"][-1].content)
    query = state["messages"][-1].content
    simulated_past_memory = "The user mentioned last week they hate rainy Mondays."

    return {"relevant_memories": simulated_past_memory}