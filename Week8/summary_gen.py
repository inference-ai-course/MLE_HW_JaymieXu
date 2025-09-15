import torch
import json
from transformers import BitsAndBytesConfig, pipeline
from pypdf import PdfReader
import os


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


def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        return text.strip()
    
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

def process_paper_to_summaries(pdf_path, llm):
    # Extract text from PDF
    paper_text = extract_text_from_pdf(pdf_path)
    
    if not paper_text:
        return None
    
    # Prepare user text format for the model
    user_text = [{"role": "user", "content": paper_text}]
    
    # Generate two different summaries with different parameters
    summary1 = generate_summary(llm, user_text, temperature=0.5, top_p=0.8)
    summary2 = generate_summary(llm, user_text, temperature=0.9, top_p=0.95)
    
    # Prepare result
    result = {
          "pdf_path": pdf_path,
          "summary_1": summary1,
          "summary_1_label": "",
          "summary_2": summary2,
          "summary_2_label": "",
      }
    
    return result


def process_all_pdfs_in_folder(folder_path, llm, save_path):
      results = []

      # Get all PDF files in the folder
      pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

      print(f"Found {len(pdf_files)} PDF files in {folder_path}")

      for i, pdf_file in enumerate(pdf_files, 1):
          pdf_path = os.path.join(folder_path, pdf_file)
          print(f"Processing {i}/{len(pdf_files)}: {pdf_file}")

          try:
              result = process_paper_to_summaries(pdf_path, llm)

              if result:
                  results.append(result)
                  
                  print(f"✓ Successfully processed: {pdf_file}")
              else:
                  print(f"✗ Failed to process: {pdf_file}")

          except Exception as e:
              print(f"✗ Error processing {pdf_file}: {e}")

      # Save all results to JSON file
      print(f"\nSaving {len(results)} results to {save_path}")
      with open(save_path, 'w', encoding='utf-8') as f:
          json.dump(results, f, indent=2, ensure_ascii=False)

      print(f"✓ Batch processing complete! Results saved to {save_path}")
      return results


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
    
    process_all_pdfs_in_folder("pdf/train/", llm, "train.json")
    process_all_pdfs_in_folder("pdf/eval/", llm, "eval.json")
    
    
if __name__ == "__main__":
    main()
