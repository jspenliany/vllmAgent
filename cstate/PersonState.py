from typing import Annotated, List, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class PersonState(TypedDict):
    # This must be Annotated with add_messages for the graph to merge history
    messages: Annotated[List[BaseMessage], add_messages]
    new_facts: List[str]
    relevant_memories: str
    intent: str  # <--- Add this field
