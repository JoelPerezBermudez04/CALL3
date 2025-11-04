from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from difflib import SequenceMatcher

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

def create_diff_html(original: str, corrected: str) -> str:
    """Create HTML highlighting the differences between original and corrected text"""
    original_words = original.split()
    corrected_words = corrected.split()
    
    matcher = SequenceMatcher(None, original_words, corrected_words)
    result_html = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result_html.append(' '.join(corrected_words[j1:j2]))
        elif tag == 'replace':
            # Word was changed
            result_html.append(f'<span class="diff-changed">{" ".join(corrected_words[j1:j2])}</span>')
        elif tag == 'insert':
            # Word was added
            result_html.append(f'<span class="diff-added">{" ".join(corrected_words[j1:j2])}</span>')
        elif tag == 'delete':
            # Word was removed (don't show in corrected text)
            pass
        result_html.append(' ')
    
    return ''.join(result_html).strip()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GrammarAssist - English Grammar Corrector</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 600px;
                width: 100%;
                padding: 60px 40px;
            }
            
            .header {
                margin-bottom: 40px;
                text-align: center;
            }
            
            .logo {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                margin-bottom: 20px;
            }
            
            .logo-icon {
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 24px;
            }
            
            .logo-text {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }
            
            h1 {
                font-size: 32px;
                color: #0f172a;
                margin-bottom: 12px;
                font-weight: 700;
            }
            
            .subtitle {
                font-size: 16px;
                color: #64748b;
                line-height: 1.5;
                margin-bottom: 8px;
            }
            
            .badge {
                display: inline-block;
                background: #e0f2fe;
                color: #0369a1;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                margin-top: 12px;
            }
            
            form {
                margin-top: 40px;
            }
            
            .form-group {
                margin-bottom: 24px;
            }
            
            label {
                display: block;
                font-size: 14px;
                font-weight: 600;
                color: #0f172a;
                margin-bottom: 10px;
            }
            
            textarea {
                width: 100%;
                padding: 16px;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                font-family: inherit;
                font-size: 16px;
                resize: none;
                transition: all 0.3s ease;
                color: #0f172a;
            }
            
            textarea:focus {
                outline: none;
                border-color: #0ea5e9;
                box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
            }
            
            textarea::placeholder {
                color: #94a3b8;
            }
            
            .button-group {
                display: flex;
                gap: 12px;
                margin-top: 32px;
            }
            
            button {
                flex: 1;
                padding: 14px 24px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
                color: white;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(14, 165, 233, 0.3);
            }
            
            .btn-primary:active {
                transform: translateY(0);
            }
            
            .features {
                margin-top: 40px;
                padding-top: 40px;
                border-top: 1px solid #e2e8f0;
            }
            
            .features-title {
                font-size: 12px;
                font-weight: 700;
                color: #94a3b8;
                text-transform: uppercase;
                margin-bottom: 16px;
                letter-spacing: 1px;
            }
            
            .features-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
            }
            
            .feature-item {
                text-align: center;
            }
            
            .feature-icon {
                font-size: 24px;
                margin-bottom: 8px;
            }
            
            .feature-label {
                font-size: 12px;
                color: #64748b;
                font-weight: 500;
            }
            
            @media (max-width: 600px) {
                .container {
                    padding: 40px 24px;
                }
                
                h1 {
                    font-size: 24px;
                }
                
                .features-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <div class="logo-icon">✓</div>
                    <div class="logo-text">GrammarAssist</div>
                </div>
                <h1>Improve Your English</h1>
                <p class="subtitle">Real-time grammar correction powered by AI. Perfect your sentences instantly.</p>
                <span class="badge">For Spanish Learners</span>
            </div>
            
            <form action="/generate_form" method="post">
                <div class="form-group">
                    <label for="text">Your Text</label>
                    <textarea 
                        id="text"
                        name="text" 
                        rows="6" 
                        placeholder="Paste your English text here. We'll check your grammar and suggest corrections..."
                        required
                    ></textarea>
                </div>
                
                <div class="button-group">
                    <button type="submit" class="btn-primary">Check Grammar</button>
                </div>
            </form>
            
            <div class="features">
                <div class="features-title">What We Check</div>
                <div class="features-grid">
                    <div class="feature-item">
                        <div class="feature-icon">📝</div>
                        <div class="feature-label">Grammar</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">✨</div>
                        <div class="feature-label">Accuracy</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">⚡</div>
                        <div class="feature-label">Instant</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">🎯</div>
                        <div class="feature-label">Precise</div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.post("/generate_form", response_class=HTMLResponse)
def generate_form(text: str = Form(...)):
    prompt = PROMPT_TEMPLATE.format(input=text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the corrected sentence from the result
    if "### Response:" in result:
        corrected = result.split("### Response:")[-1].strip()
    else:
        corrected = result
    
    diff_html = create_diff_html(text, corrected)
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Correction Result - GrammarAssist</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }}
            
            .container {{
                background: white;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 600px;
                width: 100%;
                padding: 60px 40px;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
            }}
            
            .logo {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                margin-bottom: 20px;
            }}
            
            .logo-icon {{
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 24px;
            }}
            
            .logo-text {{
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }}
            
            h1 {{
                font-size: 28px;
                color: #0f172a;
                margin-bottom: 12px;
                font-weight: 700;
            }}
            
            .status {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                color: #10b981;
                font-weight: 600;
                font-size: 16px;
                margin-top: 12px;
            }}
            
            .status-icon {{
                width: 24px;
                height: 24px;
                background: #d1fae5;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
            }}
            
            .comparison {{
                margin-top: 40px;
                padding: 24px;
                background: #f8fafc;
                border-radius: 12px;
            }}
            
            .comparison-row {{
                margin-bottom: 24px;
            }}
            
            .comparison-row:last-child {{
                margin-bottom: 0;
            }}
            
            .comparison-label {{
                font-size: 12px;
                font-weight: 700;
                color: #94a3b8;
                text-transform: uppercase;
                margin-bottom: 8px;
                letter-spacing: 1px;
            }}
            
            .comparison-text {{
                padding: 12px;
                border-radius: 8px;
                font-size: 16px;
                line-height: 1.6;
                color: #0f172a;
            }}
            
            .original {{
                background: #fee2e2;
                border-left: 4px solid #ef4444;
            }}
            
            .corrected {{
                background: #d1fae5;
                border-left: 4px solid #10b981;
            }}
            
            /* Added styles for diff highlighting */
            .diff-changed {{
                background: #fef3c7;
                color: #92400e;
                text-decoration: underline;
                text-decoration-color: #fbbf24;
                text-decoration-thickness: 2px;
                text-underline-offset: 2px;
                font-weight: 600;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            
            .diff-added {{
                background: #bbf7d0;
                color: #065f46;
                font-weight: 600;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            
            .actions {{
                display: flex;
                gap: 12px;
                margin-top: 40px;
            }}
            
            button {{
                flex: 1;
                padding: 14px 24px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }}
            
            .btn-primary {{
                background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
                color: white;
            }}
            
            .btn-primary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(14, 165, 233, 0.3);
            }}
            
            .btn-secondary {{
                background: white;
                color: #0ea5e9;
                border: 2px solid #0ea5e9;
            }}
            
            .btn-secondary:hover {{
                background: #f0f9ff;
            }}
            
            .footer {{
                text-align: center;
                margin-top: 32px;
                padding-top: 24px;
                border-top: 1px solid #e2e8f0;
            }}
            
            .footer-text {{
                font-size: 14px;
                color: #64748b;
            }}
            
            .copy-button {{
                display: inline-block;
                margin-top: 8px;
                padding: 6px 12px;
                background: #e0f2fe;
                color: #0369a1;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            
            .copy-button:hover {{
                background: #bae6fd;
            }}
            
            @media (max-width: 600px) {{
                .container {{
                    padding: 40px 24px;
                }}
                
                h1 {{
                    font-size: 24px;
                }}
                
                .actions {{
                    flex-direction: column;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <div class="logo-icon">✓</div>
                    <div class="logo-text">GrammarAssist</div>
                </div>
                <h1>Check Complete</h1>
                <div class="status">
                    <div class="status-icon">✓</div>
                    <span>Correction Complete</span>
                </div>
            </div>
            
            <div class="comparison">
                <div class="comparison-row">
                    <div class="comparison-label">Original Text</div>
                    <div class="comparison-text original">{text}</div>
                </div>
                
                <div class="comparison-row">
                    <div class="comparison-label">Corrected Text (Changes Highlighted)</div>
                    <div class="comparison-text corrected">{diff_html}</div>
                </div>
            </div>
            
            <div class="actions">
                <a href="/" style="flex: 1; text-decoration: none;">
                    <button class="btn-secondary" style="width: 100%;">← Check Another</button>
                </a>
                <button class="btn-primary copy-button" style="flex: 1;" onclick="navigator.clipboard.writeText('{corrected}'); this.textContent='Copied!'; setTimeout(() => this.textContent='Copy Result', 2000);">
                    Copy Result
                </button>
            </div>
            
            <div class="footer">
                <p class="footer-text">Keep practicing! Each correction helps you improve your English.</p>
            </div>
        </div>
    </body>
    </html>
    """
