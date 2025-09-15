import torch
from transformers import BitsAndBytesConfig, pipeline


SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are an helpful AI assistant. The user will provide you with text that has been transcribed from an audio file. Your job is to have a friendly but shy response to the content of the transcription."
}


def generate_summary(llm, user_text):
    # Combine the system prompt with the recent conversation for the model.
    prompt_context = [SYSTEM_PROMPT] + user_text

    # Use the tokenizer to apply the model's official chat template.
    prompt = llm.tokenizer.apply_chat_template(
        prompt_context,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate a response.
    outputs = llm(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    # Clean the output to get only the new assistant message.
    raw_bot_response = outputs[0]["generated_text"]
    bot_response = raw_bot_response.split("<|assistant|>")[-1].strip()

    return bot_response


def main():
    # --- LLM Loading ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    print("Loading LLM...")
    # Load LLM
    llm = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-7B-Instruct",
        model_kwargs={
            "quantization_config": quantization_config,
            "device_map": "auto",
        }
    )
    
    print("LLM loaded.")
    
    
if __name__ == "__main__":
    main()