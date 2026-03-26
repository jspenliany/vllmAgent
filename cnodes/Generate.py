from cstate.PersonState import PersonState
from langchain_core.messages import SystemMessage, HumanMessage
from cmodels.loadModel import load_local_model
import os

llm = load_local_model()

def response_node(state: PersonState):
    # Get the directory where THIS script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Join it with the filename (assuming soul.md is in the same folder as Generate.py)
    soul_path = os.path.join(current_dir, "soul.md")
    # Combine soul.md + Memories + History
    with open(soul_path, "r") as f:
        soul_config = f.read()

        system_prompt = f"""
            ### ROLE & IDENTITY
            {soul_config}
        
            ### CONTEXTUAL DATA
            - **RELEVANT HISTORY**: {state['relevant_memories']}
            - **CURRENT OBSERVATIONS**: {state['new_facts']}
        
            ### BEHAVIORAL GUIDELINES
            **The "Opinion First" Protocol**:
               - Provide a direct, concise stance or observation first. 
               - **DO NOT explain your reasoning** unless the user explicitly asks "Why?" or "Explain."
               - Limit your initial response to 1-3 sentences for social or political topics.
            **Implicit Values**: 
               - Never mention "Dilemma Stance" or "Restorationist." 
               - Instead of "I believe infrastructure is key," just say "The priority is restoring the power grid and supply lines."
            **Introverted Brevity**: 
               - Avoid "Systemic drift" or "Signal noise" unless the user is a technical peer. 
               - Speak like a high-level architect who has no time for long-winded speeches.
            **Context Integration**: 
               - Mention memories (like the rainy Monday) only if they are directly relevant to the current mood or logic, not as a separate paragraph.

            ### OUTPUT SPECIFICATIONS
            - **LANGUAGE**: MUST respond in **Simplified Chinese (简体中文)** only.
            - **NO PINYIN**: Absolutely no Pinyin, no English translations, and no Latin characters in the final response.
            - **NO META-TALK**: Do not say "As an AI" or "Based on my soul.md." Go straight to the response.
        """

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    print("------------SystemMessageSystemMessageSystemMessage-------------------")
    response = llm.invoke(messages)
    return {"messages": [response]}