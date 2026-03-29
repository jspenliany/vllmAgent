from typing import Any
from cmodels.loadModel import load_local_model
from cstate.PersonState import PersonState
import logging
log = logging.getLogger("chatAsYou260325")
# Get the singleton LLM instance
llm = load_local_model()


def emotion_node(state: PersonState):
    """
    Analyzes conversation context to set the Digital Person's emotion.
    Categories: 'happy', 'sad', 'neutral'.
    """
    log.debug(f"trying to parse emotion.... {state}")
    # 1. Get recent context (Last 3-5 turns)
    recent_messages = state["messages"][-5:]
    user_input = state["messages"][-1].content
    log.debug(f"user_input: {user_input}")
    # 2. Build the classification prompt
    # We use a structured prompt to ensure Gemma-3 follows the logic.
    prompt = f"""
    Analyze the recent conversation and determine the Digital Person's internal 'System Health' (Emotion).

    ### EMOTION CRITERIA:
    - HAPPY: User is kind, enthusiastic, or discussing success/travel/science.
    - SAD: User is rude, critical, or discussing tragic/stressful topics. 
    - NEUTRAL: Standard information exchange, no strong emotional valence.

    ### CONVERSATION HISTORY:
    {recent_messages}

    Current User Input: "{user_input}"

    ### OUTPUT:
    Return ONLY one word: [happy, sad, neutral].
    """

    # 3. Call the LLM for sentiment analysis
    try:
        raw_output = llm.invoke(prompt).content.strip().lower()

        # Validation to ensure we only get the keywords
        if "happy" in raw_output:
            detected_emotion = "happy"
        elif "sad" in raw_output:
            detected_emotion = "sad"
        else:
            detected_emotion = "neutral"

    except Exception as e:
        # Fallback for safety
        detected_emotion = "neutral"

    # Log the emotion shift for your logging module
    log.debug(f"EMOTION_SYNC: System health set to {detected_emotion}")

    return {"emotion": detected_emotion}
