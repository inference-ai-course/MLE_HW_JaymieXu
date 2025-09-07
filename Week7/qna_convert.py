import json

import settings as cfg

def convert_qa_to_synthetic():
    """Convert QnA dataset to synthetic_qa.jsonl format for fine-tuning"""
    
    system_prompt = "You are a helpful academic Q&A assistant specialized in scholarly content."
    data = []
    
    # Read the existing QnA dataset
    input_file = cfg.META / "abstract_qa_dataset.jsonl"
    output_file = cfg.PROC / "synthetic_qa.jsonl"
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            qa = json.loads(line.strip())
            user_q = qa["question"]
            assistant_a = qa["answer"]
            
            # Compose the prompt with system, user, assistant roles
            full_prompt = f"<|system|>{system_prompt}<|user|>{user_q}<|assistant|>{assistant_a}"
            data.append({"text": full_prompt})
    
    # Write to JSONL file in the PROC folder
    with open(output_file, "w", encoding="utf-8") as outfile:
        for entry in data:
            outfile.write(json.dumps(entry) + "\n")
    
    print(f"Converted {len(data)} QA pairs to synthetic_qa.jsonl format")
    print(f"Output saved to: {output_file}")
    
    return len(data)

if __name__ == "__main__":
    convert_qa_to_synthetic()