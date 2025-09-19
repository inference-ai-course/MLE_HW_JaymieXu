from pathlib import Path

from transformers import pipeline

class Summarizer:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        print("ASR model with size {in_model_size.value} loaded With {in_compute_type.value}")


    def __del__(self):
        del self.summarizer
        
        
    def summarize(self, text, in_max_length, in_min_length = 30, in_do_sample = False):
        return self.summarizer(text, max_length=130, min_length=in_min_length, do_sample=in_do_sample)