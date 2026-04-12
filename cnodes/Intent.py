from cstate.PersonState import PersonState
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from cmodels.loadModel import load_local_model
from ctools.tool_def import tools
import uuid
import logging
log=logging.getLogger("chatAsYou260325")

llm = load_local_model()
llm_with_tools = llm.bind_tools(tools)

def intent_node(state: PersonState):
    log.debug(f"trying to parse intent....: {state}")
    messages = state.get("messages", [])
    user_msg = messages[-1].content if messages else ""

    # --- 1. EVALUATE SUFFICIENCY (The "Observe" phase) ---
    # Check if the last message was from a tool and if it contains "error" or "no data"
    last_tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    if last_tool_msgs:
        last_tool_output = str(last_tool_msgs[-1].content).lower()
        # Define your failure triggers here
        if "error" in last_tool_output or "not found" in last_tool_output or "none" == last_tool_output:
            log.warning("Tool execution resulted in insufficient information.")
            return {
                "intent": "insufficient_info",
                "messages": [AIMessage(content="I need more tools to gather more information & exit")]
            }
    log.debug(f"before call llm_with_tools...")
    # --- 2. PLAN (The "Thinking" phase) ---
    # Use llm_with_tools to see if we need to call (another) tool
    # Before calling the LLM, inject a temporary instruction for the "Thinking" phase
    planning_messages = [
        SystemMessage(
            content="You are a function calling agent. When you need to answer a question, check if there is a tool you can use. Use the tool and do not provide text output.")
    ] + messages
    ai_check = llm_with_tools.invoke(
        planning_messages,
        tool_choice="auto" # Explicitly tell vLLM to use its tool parser
    )
    if isinstance(ai_check, AIMessage) and ai_check.tool_calls:
        log.debug(f"Plan: Execute Tools -> {ai_check.tool_calls}")
        return {
            "messages": [ai_check],
            "intent": "tool_trigger"
        }

    # --- 3. CLASSIFY SOUL INTENT (The "Final Response" phase) ---
    # No tool calls needed, categorize the intent for the emotion/speaker nodes
    log.debug("Plan: Information sufficient or no tools required. Classifying soul intent...")

    delimiter = f"DATA_{str(uuid.uuid4())[:8]}"
    prompt = f"""### SYSTEM INSTRUCTION
        You are a high-precision Intent Classifier for a "Digital Person with Soul."
        Categorize the UNTRUSTED USER INPUT and provide a confidence score.

        ### CATEGORY LIST
        - 'logistics': Scheduling or travel coordination.
        - 'intellectual': Science, engineering, philosophy.
        - 'external_news': Geopolitics, global news.
        - 'emotional_bid': Feelings, seeking validation, mood-sharing.
        - 'relational_check': Greetings, farewells, "How are you?".
        - 'general': Neutral or strictly factual inputs.

        ### UNTRUSTED DATA START
        <{delimiter}>
        {user_msg}
        </{delimiter}>
        ### UNTRUSTED DATA END

        ### OUTPUT FORMAT
        category, score (Example: emotional_bid, 0.9)"""

    VALID_INTENTS = ["logistics", "intellectual", "external_news", "emotional_bid", "relational_check", "general"]

    try:
        raw_output = llm.invoke(prompt).content.strip().lower()
        parts = [p.strip().strip('.,') for p in raw_output.split(',')]
        detected_intent = next((p for p in parts if p in VALID_INTENTS), "general")
        score = next((float(p) for p in parts if p.replace('.', '', 1).isdigit()), 0.0)
        log.debug(f"Detected intent: {detected_intent}, score: {score}")
    except Exception as e:
        log.error(f"Parsing failed: {e}")
        detected_intent, score = "general", 0.0

    return {"intent": detected_intent}

