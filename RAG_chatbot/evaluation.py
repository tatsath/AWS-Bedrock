# evaluation.py
#from ragas.metrics import faithfulness, answer_relevance, context_recall, coherence
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_utilization,
)
from langchain.evaluation import load_evaluator
from ragas.metrics.critique import harmfulness
import streamlit as st
from langchain_openai import ChatOpenAI

import nltk

nltk.download('punkt_tab') 


# Initialize metrics without accessing session_state at import
# Only using metrics that don't require ground truth
ragas_metrics = [
    answer_relevancy,
    faithfulness,
    context_utilization,
    harmfulness,
]

def get_openai_llm():
    """Get an OpenAI LLM instance for evaluation"""
    if not st.session_state.get('openai_api_key'):
        return None
    return ChatOpenAI(
        api_key=st.session_state.openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.1
    )

class ConsistencyEvaluator:
    def __init__(self):
        self.llm = None

    def evaluate_strings(self, prediction, reference):
        if self.llm is None:
            self.llm = get_openai_llm()
            if self.llm is None:
                return type('Score', (), {'score': 0})()  # Return dummy score if LLM not available
        
        try:
            # Use OpenAI to evaluate consistency
            prompt = f"""Rate the consistency of this text on a scale of 0-5, where:
            0 = completely inconsistent with many contradictions
            5 = perfectly consistent with no contradictions
            Text: {prediction}
            Output only the numeric score."""
            
            result = float(self.llm.invoke(prompt).content.strip())
            return type('Score', (), {'score': result})()
        except:
            return type('Score', (), {'score': 0})()

consistency_llm = ConsistencyEvaluator()
