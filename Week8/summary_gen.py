import torch
from transformers import BitsAndBytesConfig, pipeline


SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a scientific editor, write a 220-250 word, strictly faithful summary for grad-level ML readers stating the problem, core method (plain language), data/setting, key quantitative results, novelty vs prior work, and limitations; use only the provided text (no speculation)."
}


def generate_summary(llm, user_text, temperature=0.7, top_p=0.95, top_k=50, max_new_tokens=250):        
      # Combine the system prompt with the recent conversation for the model.
      prompt_context = [SYSTEM_PROMPT] + user_text

      # Use the tokenizer to apply the model's official chat template.
      prompt = llm.tokenizer.apply_chat_template(
          prompt_context,
          tokenize=False,
          add_generation_prompt=True
      )

      # Generate a response.
      outputs = llm(prompt,
                   max_new_tokens=max_new_tokens,
                   do_sample=True,
                   temperature=temperature,
                   top_k=top_k,
                   top_p=top_p)

      # Clean the output to get only the new assistant message.
      raw_bot_response = outputs[0]["generated_text"]
      bot_response = raw_bot_response.split("<|assistant|>")[-1].strip()

      return bot_response


def main():
    # --- LLM Loading ---
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
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