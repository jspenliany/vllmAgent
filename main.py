from typing import Any, Sequence, cast
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from cstate.PersonState import PersonState
from cnodes.ExtractDetail import extract_memory_node
from cnodes.RetrieveDetail import retrieve_memory_node
from cnodes.IntentGenerate import response_node
from cnodes.Intent import intent_node
from cnodes.Emotion import emotion_node
from ctools.tool_def import tools
from logger.log_def import setup_singleton_logger
import uuid
#start logging system
log=setup_singleton_logger()
# 1. Define your tools and the ToolNode
tool_node = ToolNode(tools)

# 2. Define the Routing Logic (The "Plan" and "Observe" decision)
def route_after_intent(state: PersonState):
    """
    :param state:
    :return:
    """
    log.debug("--- Routing after Intent ---")

    last_msg = state["messages"][-1]

    # 1. Check if the LLM generated a tool call (The ReAct "Plan")
    # We use isinstance to avoid the AttributeError on HumanMessages
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        log.debug("Plan: Execute Tool")
        return "tools"

    # 2. Re-evaluating the "Soul" path
    # If the user input was general, we might need more "thought" (reflector)
    if state.get("intent") == "general":
        log.debug("Plan: Reflecting more")
        return "reflector"

    # 3. Default: Move to feeling (emotion) and responding
    log.debug("Plan: Proceed to Emotion")
    return "emotion_tracker"


def run_digital_person(name):
    log.debug("loading workflow.......")
    workflow = StateGraph(PersonState)

    # Define the flow
    workflow.add_node("listener", extract_memory_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("reflector", retrieve_memory_node)
    workflow.add_node("intent_classifier", intent_node)
    workflow.add_node("emotion_tracker", emotion_node)
    workflow.add_node("speaker", response_node)
    log.debug("all nodes loaded.............")
    # Connect them
    workflow.add_edge(START, "listener")
    workflow.add_edge("listener", "reflector")
    workflow.add_edge("reflector", "intent_classifier")  # Route to Intent
    # workflow.add_edge("intent_classifier", "emotion_tracker")  # Route to Emotion
    workflow.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "reflector":"reflector",
            "emotion_tracker":"emotion_tracker",
        }
    )
    workflow.add_edge("tool_node", "intent_classifier")
    # Route to Emotion
    workflow.add_edge("emotion_tracker", "speaker")  # Then to Speaker
    workflow.add_edge("speaker", END)
    log.debug("workflow loaded.............")
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # Use a fixed ID or move uuid inside the loop if you want fresh starts
    session_id = str(uuid.uuid4())
    # Define the config with an explicit type
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    log.debug(f"--- 虚拟形象已上线 (Thread: {session_id}) ---")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        # Assuming your state class is named PersonState
        inputs = cast(PersonState, cast(Any, {"messages": [HumanMessage(content=user_input)]}))

        for event in app.stream(inputs, config=config, stream_mode="values"):
            # log.debug("-----------***********--------------")
            if "messages" in event:
                # The last message in the list is the most recent (Human or AI)
                last_msg = event["messages"][-1]

                # Only print if it's from the Digital Person (AIMessage)
                if isinstance(last_msg, AIMessage):
                    log.debug(f"Digital Person: {last_msg.content}")
                    print(f"Digital Person: {last_msg.content}")
                    log.debug("---------------------------------")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_digital_person('PyCharm')

