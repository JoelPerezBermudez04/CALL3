from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

PROMPT_TEMPLATE = """### Instruction:
Correct this sentence. (just give me the corrected sentence, no explanations needed)

### Input:
{input}

### Response:
"""


app = FastAPI()

base_model_name = "unsloth/gemma-2-2b-it-bnb-4bit"
lora_path = "grammar_checker_lora"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Modelo base 4-bit en CPU
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map={"": "cpu"},
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    torch_dtype=torch.float16
)

# Aplicar LoRA en CPU
model = PeftModel.from_pretrained(base_model, lora_path, device_map={"": "cpu"})

# Página principal con formulario
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Generador de texto</title>
        </head>
        <body>
            <h1>Escribe tu texto:</h1>
            <form action="/generate_form" method="post">
                <textarea name="text" rows="4" cols="50"></textarea><br><br>
                <input type="submit" value="Generar">
            </form>
        </body>
    </html>
    """

# Endpoint para recibir el formulario
@app.post("/generate_form", response_class=HTMLResponse)
def generate_form(text: str = Form(...)):
    prompt = PROMPT_TEMPLATE.format(input=text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return f"""
    <html>
        <head><title>Resultado</title></head>
        <body>
            <h1>Texto generado:</h1>
            <p>{result}</p>
            <a href="/">Volver</a>
        </body>
    </html>
    """
