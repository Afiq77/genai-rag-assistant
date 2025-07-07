# model_loader.py

"""
Placeholder for loading local or cloud LLMs (e.g., OpenAI, Ollama, Mistral).
This can be expanded as needed based on your deployment choice.
"""

from langchain.llms import HuggingFaceHub

def load_llm():
    # Example: Using Hugging Face Hub model
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.3, "max_length": 256}
    )
    return llm
