import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, request, jsonify, Response

MODEL_DIR = os.environ.get("MODEL_DIR", "erdemKocaogluu/byt5-small-tr-normalizer")
PREFIX    = "düzelt: "
MAXLEN    = 384
BEAM      = 1

print(f"Loading model: {MODEL_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
print(f"Model ready (device={device}).")


def correct(sentence: str) -> str:
    sentence = (sentence or "").strip()
    if not sentence:
        return ""
    enc = tokenizer(PREFIX + sentence, return_tensors="pt",
                    truncation=True, max_length=MAXLEN).to(device)
    with torch.no_grad():
        output = model.generate(**enc, num_beams=BEAM, max_length=MAXLEN)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Turkish Text Normalizer</title>
<style>
 :root{--bg:#f1f5f9;--card:#fff;--border:#e2e8f0;--blue:#2563eb;--dark:#0f172a;--gray:#64748b}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--dark);
   font-family:system-ui,Segoe UI,Arial,sans-serif;min-height:100vh;display:flex;
   align-items:flex-start;justify-content:center;padding:48px 16px}
 .box{background:var(--card);border:1px solid var(--border);border-radius:16px;
   padding:28px;max-width:640px;width:100%;box-shadow:0 6px 24px rgba(0,0,0,.06)}
 h1{margin:0 0 4px;font-size:22px} p.subtitle{margin:0 0 20px;color:var(--gray);font-size:14px}
 label{display:block;font-size:13px;color:var(--gray);margin:14px 0 6px}
 textarea{width:100%;border:1px solid var(--border);border-radius:10px;padding:12px;
   font-size:16px;font-family:inherit;resize:vertical;min-height:80px}
 button{margin-top:16px;width:100%;background:var(--blue);color:#fff;border:0;
   border-radius:10px;padding:13px;font-size:16px;font-weight:600;cursor:pointer}
 button:disabled{opacity:.6;cursor:default}
 .output{margin-top:18px;background:#f8fafc;border:1px solid var(--border);border-radius:10px;
   padding:14px;min-height:54px;font-size:16px;white-space:pre-wrap}
 .output:empty::before{content:"The corrected sentence will appear here";color:#94a3b8}
</style></head>
<body><div class="box">
  <h1>Turkish Text Normalizer</h1>
  <p class="subtitle">Rewrites informal, dialectal, or misspelled Turkish sentences into standard Turkish.</p>
  <label for="g">Sentence</label>
  <textarea id="g" placeholder="e.g. sicil belgmi alcam"></textarea>
  <button id="b" onclick="correct()">Correct</button>
  <label>Corrected</label>
  <div id="c" class="output"></div>
</div>
<script>
async function correct(){
  const g=document.getElementById('g').value, b=document.getElementById('b'), c=document.getElementById('c');
  if(!g.trim()){c.textContent='';return;}
  b.disabled=true; b.textContent='Correcting...'; c.textContent='';
  try{
    const r=await fetch('/correct',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sentence:g})});
    const d=await r.json(); c.textContent=d.output || '(empty)';
  }catch(e){ c.textContent='Error: '+e; }
  b.disabled=false; b.textContent='Correct';
}
document.getElementById('g').addEventListener('keydown',e=>{
  if((e.ctrlKey||e.metaKey)&&e.key==='Enter') correct();
});
</script></body></html>"""


app = Flask(__name__)

@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")

@app.route("/correct", methods=["POST"])
def correct_api():
    data = request.get_json(silent=True) or {}
    return jsonify({"output": correct(data.get("sentence", ""))})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting server: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port)
