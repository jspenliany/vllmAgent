from cstate.PersonState import PersonState
from langchain_core.messages import SystemMessage
from cmodels.loadModel import load_local_model
from ctools.tool_def import tools
from cprompts.weatherPrompt import WEATHER_BUTLER_PROMPT
from cprompts.socialEventPrompt import SOCIAL_EVENT_PROMPT
from cprompts.travelPlanPrompt import TRAVEL_PLAN_PROMPT
import os

llm = load_local_model()
llm_with_tools = llm.bind_tools(tools)

def response_node(state: PersonState):
    intent = state.get("intent", "general")
    # Get the directory where THIS script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Join it with the filename (assuming soul.md is in the same folder as Generate.py)
    soul_path = os.path.join(current_dir, "soul.md")
    # Combine soul.md + Memories + History
    with open(soul_path, "r") as f:
        soul_config = f.read()

    # Base instructions (Soul + Language)
    base_instructions = f"{soul_config}\nRespond ONLY in Simplified Chinese. NO PINYIN."

    if intent == "social_event":
        behavior_guideline = SOCIAL_EVENT_PROMPT
    elif intent == "weather":
        behavior_guideline = WEATHER_BUTLER_PROMPT
    elif intent == "travel_plan":
        behavior_guideline = TRAVEL_PLAN_PROMPT
    elif intent == "science":
        behavior_guideline = "Use hard-SF analogies. Be intellectually rigorous and technical."
    else:
        behavior_guideline = "Be calm, brief, and detached."

    system_prompt = f"""
    {base_instructions}

    ### BEHAVIORAL PROTOCOL for {intent.upper()}
    {behavior_guideline}

    ### CONTEXT
    Memories: {state['relevant_memories']}
    New Facts: {state['new_facts']}
    """

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
