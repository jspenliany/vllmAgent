from cstate.PersonState import PersonState
from langchain_core.messages import HumanMessage
from cmodels.loadModel import load_local_model
from logger.log_def import setup_singleton_logger
log=setup_singleton_logger()

llm = load_local_model()

def extract_memory_node(state: PersonState):
    log.info("Extracting Memory Node.....begin")
    last_user_msg = state["messages"][-1].content

    prompt = f"""Analyze this message: '{last_user_msg}'
    Did the user share a life detail (preference, habit, history, emotion)?
    If yes, extract it as a short fact. If no, return 'None'.
    Format: [Fact 1, Fact 2]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    # In a real app, you'd save state['new_facts'] to a Vector DB here
    log.info(response)
    return {"new_facts": [response.content]}