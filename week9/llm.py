import json
from pathlib import Path
from transformers import pipeline
from summarize import Summarizer
from search import Search

SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a helpful AI assistant with access to tools. The user will provide you with text that has been transcribed from an audio file. Your job is to have a friendly but shy response to the content of the transcription.

    You have access to the following tools:
    1. search_arxiv(query) - for searching arXiv papers
    2. notion() - for sending content to Notion

    When the user asks to search arXiv papers or research, respond with a JSON function call in this exact format:
    {"function": "search_arxiv", "arguments": {"query": "the search query"}}

    When the user asks to create a Notion note, respond with a JSON function call in this exact format:
    {"function": "notion", "arguments": {}}

    For all other conversations, respond normally with friendly text (no JSON).

    Examples:
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
        
        print("LLM loaded.")
        
        self.rag_search = Search()
        self.summarizer = Summarizer()
        
        
    def route_llm_output(self, llm_output: str) -> str:
        """
        Route LLM response to the correct tool if it's a function call, else return the text.
        Expects LLM output in JSON format like {'function': ..., 'arguments': {...}}.
        """
        try:
            output = json.loads(llm_output)
            func_name = output.get("function")
            args = output.get("arguments", {})
            
        except (json.JSONDecodeError, TypeError):
            # Not a JSON function call; return the text directly
            return llm_output

        if func_name == "search_arxiv":
            query = args.get("query", "")
            print(f"FUNCTION CALL: search_arxiv(query='{query}')")
            
            # Get top 3 rag paper and combine
            rag_results = self.rag_search.search(query, 3)
            combined_text = ""
            for search_hit in rag_results:
                combined_text += search_hit.text + "\n"
            
            # Sum
            result = self.summarizer.summarize(combined_text, 30)
            
            print(f"FUNCTION OUTPUT: {result}")
            return f"Using tool: {result}"
        
        elif func_name == "notion":
            print(f"FUNCTION CALL: notion()")
            # TODO: Add notion functionality
            result = llm_output
            print(f"FUNCTION OUTPUT: {result}")
            return f"Using tool: {result}"
        
        else:
            print(f"UNKNOWN FUNCTION: {func_name}")
            return f"Error: Unknown function '{func_name}'"
        
        
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
                       max_new_tokens=160,
                       do_sample=False,
                       temperature=0.6,
                       top_p=0.9,
                       repetition_penalty=1.05,
                       return_full_text=False,
                       eos_token_id=eos_ids)
        
        # Get only the generated response
        bot = out[0]["generated_text"].strip()
        
        output = self.route_llm_output(bot)
        
        self.conversation_history.append({"role":"assistant","content":output})
        
        return output