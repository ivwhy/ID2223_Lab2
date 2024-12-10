from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import gradio as gr
import torch


base_model = "unsloth/Llama-3.2-3B-Instruct"  # Replace with the correct base model
peft_model_path = "ivwhy/lora_model"

config = PeftConfig.from_pretrained(peft_model_path)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, peft_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # CPU
)

chatbot = pipeline

message_list = []
response_list = []

def chat_function(message, history, system_prompt, max_new_tokens, temperature):
    messages = [{"role":"system","content":system_prompt},
                {"role":"user","content":message}]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,)
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(
        prompt,
        max_new_tokens = max_new_tokens,
        eos_token_id = terminators,
        do_sample = True,
        temperature = 0.1,
        top_p = 0.9,)
    return outputs[0]["generated_text"][len(prompt):]

demo_chatbot = gr.ChatInterface(
    chat_function,
    textbox=gr.Textbox(placeholder="Enter message here", container=False, scale=7),
    chatbot=gr.Chatbot(height=400),
    additional_inputs=[
        gr.Textbox("You are helpful AI", label="System Prompt"),
        gr.Slider(500,4000, label="Max New Tokens"),
        gr.Slider(0,1,label="Temperature")
    ])

demo_chatbot.launch()

''' =================================== OLD VERSION ==============================================
import torch
import transformers
import gradio as gr
from unsloth import FastLanguageModel

# Load the fine-tuned Unsloth model
max_seq_length = 2048  # Adjust based on your training
dtype = None  # Auto-detect is fine for CPU

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="ivwhy/lora_model",  # Your fine-tuned model path
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,  # Keep 4-bit loading enabled
    )

    # Optional: Add special tokens for chat if needed
    tokenizer.pad_token = tokenizer.eos_token

    # Create the pipeline for CPU
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # Force CPU usage
    )
    
    return pipeline, tokenizer

# Load model globally
generation_pipeline, tokenizer = load_model()

def chat_function(message, history, system_prompt, max_new_tokens, temperature):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Define terminators
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # Generate response
    outputs = generation_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    
    # Extract and return just the generated text
    return outputs[0]["generated_text"][len(prompt):]

# Create Gradio interface
demo = gr.ChatInterface(
    chat_function,
    textbox=gr.Textbox(placeholder="Enter message here", container=False, scale=7),
    chatbot=gr.Chatbot(height=400),
    additional_inputs=[
        gr.Textbox("You are helpful AI", label="System Prompt"),
        gr.Slider(minimum=1, maximum=4000, value=500, label="Max New Tokens"),
        gr.Slider(minimum=0, maximum=1, value=0.7, label="Temperature")
    ]
)

if __name__ == "__main__":
    demo.launch()

================================== OLD VER ==============================
import torch
import transformers
import gradio as gr
from unsloth import FastLanguageModel

# Load the fine-tuned Unsloth model
max_seq_length = 2048  # Adjust based on your training
dtype = None  # None for auto detection

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="ivwhy/lora_model",  # Your fine-tuned model path
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True  # Optional: load in 4-bit for efficiency
    )

    # Optional: Add special tokens for chat if needed
    tokenizer.pad_token = tokenizer.eos_token

    # Create the pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1  # Use GPU if available
    )
    
    return pipeline, tokenizer

# Load model globally
generation_pipeline, tokenizer = load_model()

def chat_function(message, history, system_prompt, max_new_tokens, temperature):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Define terminators
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # Generate response
    outputs = generation_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    
    # Extract and return just the generated text
    return outputs[0]["generated_text"][len(prompt):]

# Create Gradio interface
demo = gr.ChatInterface(
    chat_function,
    textbox=gr.Textbox(placeholder="Enter message here", container=False, scale=7),
    chatbot=gr.Chatbot(height=400),
    additional_inputs=[
        gr.Textbox("You are helpful AI", label="System Prompt"),
        gr.Slider(minimum=1, maximum=4000, value=500, label="Max New Tokens"),
        gr.Slider(minimum=0, maximum=1, value=0.7, label="Temperature")
    ]
)

if __name__ == "__main__":
    demo.launch()

'''