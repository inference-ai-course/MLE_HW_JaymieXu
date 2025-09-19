from pathlib import Path
from transformers import pipeline

SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a helpful AI assistant with access to tools. The user will provide you with text that has been transcribed from an audio file. Your job is to have a friendly but shy response to the content of the transcription.

    You have access to the following tools:
    1. summarize() - for summarizing the transcription
    2. search_arxiv(query) - for searching arXiv papers
    3. notion() - for sending content to Notion

    When the user asks to summarize, respond with a JSON function call in this exact format:
    {"function": "summarize", "arguments": {}}

    When the user asks to search arXiv papers or research, respond with a JSON function call in this exact format:
    {"function": "search_arxiv", "arguments": {"query": "the search query"}}

    When the user asks to create a Notion note, respond with a JSON function call in this exact format:
    {"function": "notion", "arguments": {}}

    For all other conversations, respond normally with friendly text (no JSON).

    Examples:
    User: "Can you summarize this?"
    Assistant: {"function": "summarize", "arguments": {}}

    User: "Search for papers about quantum computing"
    Assistant: {"function": "search_arxiv", "arguments": {"query": "quantum computing"}}

    User: "Save this to Notion"
    Assistant: {"function": "notion", "arguments": {}}

    User: "How are you today?"
    Assistant: Oh, hello! I'm doing well, thank you for asking. How are you?"""
}

class LLM:
    def __init__(self):
        self.conversation_history = []
        
        self.llm = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-3B-Instruct",
            model_kwargs={
                "device_map": "auto"
            }
        )
        
        
    def generate_response(self, user_text):
        # Get the last 5 turns of the conversation.
        self.conversation_history.append({"role":"user","content":user_text})
        
        # Frame the user's message to give context to the model.
        prompt = self.llm.tokenizer.apply_chat_template(
            [SYSTEM_PROMPT] + self.conversation_history[-10:], tokenize=False, add_generation_prompt=True
        )
        
        tok = self.llm.tokenizer
        
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        
        # End token
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [tok.eos_token_id] + ([im_end] if im_end is not None else [])
        
        out = self.llm(prompt,
                       max_new_tokens=100,
                       do_sample=True,
                       temperature=0.7,
                       top_k=50,
                       top_p=0.95,
                       return_full_text=False,
                       eos_token_id=eos_ids)
        
        # Get only the generated response
        bot = out[0]["generated_text"].strip()
        self.conversation_history.append({"role":"assistant","content":bot})
        
        return bot