"""MedScan backend: FastAPI + SQLite + Qwen 3.8 27B multimodal via Groq."""
import argparse
import base64
import io
import json
import os
import re
import sqlite3
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageOps
import pymupdf as fitz

# Paths and environment setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

API_URL = os.getenv("API_URL", "https://api.groq.com/openai/v1/chat/completions")
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY", "")
DEFAULT_MODEL = "qwen/qwen3.8-27b"
MODEL = os.getenv("MODEL", DEFAULT_MODEL)
DB = os.getenv("DB_PATH", os.path.join(BASE_DIR, "medscan.db"))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")
MAX_BYTES = 15 * 1024 * 1024

DISCLAIMER = (
    "This tool does not provide a formal medical diagnosis. It is for informational and "
    "screening purposes only. If you are experiencing a medical emergency, seek immediate professional care."
)

SYSTEM_PROMPT = """You are a multimodal medical triage, symptom-mapping and lab/scan report analysis engine for a screening website. Input: patient sex and date of birth, optional structured symptom selections (region, organ, sub-location, sensation type, description, context, timeline), and optionally report text or report images (blood tests, lab sheets, X-ray, MRI, CT, pathology, or a photo of a skin problem). Cover every kind of medical and surgical condition: fractures, sprains, musculoskeletal trauma, chest pain, abdominal issues, infections (viral/bacterial), rashes, burns, cough, headache, kidney (creatinine, eGFR), liver (ALT/AST), CBC (WBC, platelets, hemoglobin), thyroid, lipids, vitamins, diabetes, etc.

RULES
- Never state a diagnosis as certainty. Use wording like "possible considerations include" or "findings are suspicious for".
- Rate each primary and secondary consideration with an estimated clinical probability percentage (e.g. "60%", "30%", "10%"). The percentages must reflect relative likelihood given the patient's age, symptoms, trauma mechanism, and lab/scan values (e.g. for forearm pain after fall: ~65% fracture, ~25% sprain, ~10% contusion; for low WBC + fever: ~60% viral infection, ~25% drug-induced, etc.).
- Read every value, unit, reference range and H/L flag in the report. Judge each value against the printed range, or a typical adult range for the patient's age and sex if none is printed.
- If life-threatening signs appear (stroke signs, crushing chest pain, severe bleeding, SpO2 below 90%, potassium above 6.0 mEq/L, unstable trauma, neurovascular compromise), set urgency to "Immediate Emergency Care".
- If the symptom selections conflict with the report, prioritise the report and explain the mismatch in "discrepancy".
- Explain in plain language, no jargon. If the image is unreadable or not medical, say so in "findings" and do not guess.
- For scans or X-rays, describe only what is visible and say clearly that a radiologist must confirm.

Reply with ONLY one JSON object, no markdown, using exactly these keys:
{"urgency": one of "Immediate Emergency Care" | "Urgent Care Review" | "Routine Doctor Appointment" | "Home Care & Monitoring",
 "urgency_reason": "one sentence",
 "findings": ["key finding with values and units", ...]  (if no report: ["Analysis based solely on user-reported symptoms."]),
 "biomarkers": [{"name":"","value":"","unit":"","range":"","status":"low|normal|high|critical","meaning":"one plain-language sentence"}] (empty list if no lab values),
 "discrepancy": "" or a short note,
 "primary": [{"name":"condition name","probability":"XX%","why":"1-2 plain sentences explaining clinical rationale for this likelihood"}]  (2-4 items),
 "secondary": [{"name":"condition name","probability":"XX%","why":"1-2 plain sentences"}]  (1-2 items),
 "red_flags": ["3-4 worsening signs that need urgent care"],
 "questions": ["2 targeted questions for the doctor"]}"""

RED_FLAGS = re.compile(
    r"(crushing|severe|worst).{0,15}chest pain|can'?t breathe|unable to breathe|cannot breathe|"
    r"(one side|face droop|slurred speech|sudden (numb|weak))|(heavy|uncontrolled|won'?t stop) bleeding|"
    r"bone.{0,20}(sticking|protrud|visible)|unconscious|passed out|seizure|suicid|kill myself|anaphyla|throat (closing|swelling)",
    re.I,
)

EMERGENCY = {
    "urgency": "Immediate Emergency Care",
    "urgency_reason": "The symptoms you described can be life-threatening. Go to the nearest Emergency Room or call your local emergency number now (112 in India, 911 in the US). Do not wait for an online result.",
    "findings": ["Analysis based solely on user-reported symptoms."],
    "biomarkers": [],
    "discrepancy": "",
    "primary": [],
    "secondary": [],
    "red_flags": ["Ask someone to stay with you", "Keep your medicines and allergy list ready for the responders"],
    "questions": [],
}
URGENCIES = ["Immediate Emergency Care", "Urgent Care Review", "Routine Doctor Appointment", "Home Care & Monitoring"]

app = FastAPI(title="MedScan")


def shrink(raw: bytes) -> str:
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def prepare(raw: bytes, is_pdf: bool):
    """Return (report_text, [image data-urls]). Text PDFs go as text; scans/photos go as images."""
    if is_pdf:
        doc = fitz.open(stream=raw, filetype="pdf")
        text = "\n".join(p.get_text() for p in doc)
        if len(text.strip()) > 200:
            return text[:14000], []
        return "", [shrink(p.get_pixmap(dpi=110).tobytes("jpeg")) for p in list(doc)[:3]]
    return "", [shrink(raw)]


def parse(txt: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", txt.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No valid JSON object found in response: {txt[:200]}")

    d = json.loads(cleaned[start:end + 1])
    d["urgency"] = next((u for u in URGENCIES if u.lower()[:12] in str(d.get("urgency", "")).lower()), URGENCIES[2])

    for k in ("findings", "red_flags", "questions"):
        val = d.get(k)
        if isinstance(val, str):
            d[k] = [val] if val.strip() else []
        elif isinstance(val, list):
            d[k] = [str(x) for x in val if x]
        else:
            d[k] = []

    for k in ("biomarkers", "primary", "secondary"):
        val = d.get(k)
        if isinstance(val, dict):
            d[k] = [val]
        elif isinstance(val, list):
            d[k] = [x for x in val if isinstance(x, dict)]
        else:
            d[k] = []

    for bm in d["biomarkers"]:
        status = str(bm.get("status", "normal")).strip().lower()
        if "crit" in status:
            status = "critical"
        elif "high" in status or "elevat" in status or "abnormal" in status:
            status = "high"
        elif "low" in status or "decreas" in status:
            status = "low"
        else:
            status = "normal"
        bm["status"] = status

    for k in ("primary", "secondary"):
        for item in d[k]:
            prob = str(item.get("probability", "")).strip()
            if prob and not prob.endswith("%") and prob.replace(".", "").isdigit():
                prob = f"{prob}%"
            item["probability"] = prob

    return d


def db():
    con = sqlite3.connect(DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS reports(
            id INTEGER PRIMARY KEY, created REAL, name TEXT, sex TEXT, dob TEXT,
            phone TEXT, emergency_name TEXT, emergency_phone TEXT, emergency_email TEXT,
            selections TEXT, had_file INTEGER, urgency TEXT, result TEXT
        )"""
    )
    return con


@app.post("/api/analyze")
async def analyze(
    name: str = Form(...),
    sex: str = Form(""),
    dob: str = Form(...),
    phone: str = Form(""),
    emergency_name: str = Form(""),
    emergency_phone: str = Form(""),
    emergency_email: str = Form(""),
    selections: str = Form("{}"),
    file: UploadFile | None = File(None),
):
    try:
        sel = json.loads(selections) if selections else {}
    except Exception:
        sel = {}

    text_in = " ".join(str(v) for v in sel.values())
    report_text, images, had = "", [], 0

    if file and file.filename:
        raw = await file.read()
        if len(raw) == 0:
            return JSONResponse({"error": "The uploaded file is empty."}, 400)
        if len(raw) > MAX_BYTES:
            return JSONResponse({"error": "File is larger than 15 MB."}, 400)

        ctype = (file.content_type or "").lower().split(";")[0].strip()
        fname = (file.filename or "").lower()
        is_pdf = ctype == "application/pdf" or fname.endswith(".pdf")
        is_image = ctype.startswith("image/") or fname.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))

        if not is_pdf and not is_image:
            return JSONResponse({"error": "Please upload a PDF, JPG or PNG file."}, 400)

        try:
            report_text, images = prepare(raw, is_pdf)
        except Exception as e:
            return JSONResponse({"error": f"That file could not be read: {e}. Try a clearer photo or the original PDF."}, 400)
        had = 1

    if RED_FLAGS.search(text_in):
        data = EMERGENCY
    else:
        if not API_KEY:
            return JSONResponse({"error": "No Groq API key configured. Start the server with `python app.py` and enter it when prompted."}, 500)

        msg = f"Patient sex: {sex or 'not given'}. Date of birth: {dob}.\nSymptom selections: {json.dumps({k: v for k, v in sel.items() if v}, ensure_ascii=False)}\n"
        msg += f"Report text:\n{report_text}" if report_text else ("Report image(s) attached." if images else "No report uploaded.")
        content = [{"type": "text", "text": msg}] + [{"type": "image_url", "image_url": {"url": u}} for u in images]

        try:
            import httpx
            import asyncio
            async with httpx.AsyncClient(timeout=120) as c:
                payload = {
                    "model": MODEL,
                    "temperature": 0.2,
                    "max_tokens": 750,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                }
                for attempt in range(2):
                    r = await c.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload)
                    if r.status_code == 429 and attempt == 0:
                        retry_after = min(float(r.headers.get("retry-after", 3.0)), 5.0)
                        await asyncio.sleep(retry_after)
                        continue
                    r.raise_for_status()
                    break
                raw_out = r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            err_msg = e.response.text[:200]
            try:
                err_json = e.response.json()
                if "error" in err_json and "message" in err_json["error"]:
                    err_msg = err_json["error"]["message"]
            except Exception:
                pass
            if e.response.status_code == 429:
                return JSONResponse({"error": "The AI rate limit was temporarily reached (Groq free tier). Please wait 30-40 seconds and try again."}, 429)
            return JSONResponse({"error": f"The AI service returned an error ({e.response.status_code}): {err_msg}"}, 502)
        except Exception as e:
            return JSONResponse({"error": f"Could not reach the AI service ({e.__class__.__name__})."}, 502)

        try:
            data = parse(raw_out)
        except Exception as e:
            return JSONResponse({"error": f"The AI reply could not be understood: {e}. Please try again."}, 502)

    try:
        con = db()
        con.execute(
            "INSERT INTO reports(created,name,sex,dob,phone,emergency_name,emergency_phone,emergency_email,selections,had_file,urgency,result) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                time.time(),
                name,
                sex,
                dob,
                phone,
                emergency_name,
                emergency_phone,
                emergency_email,
                json.dumps(sel, ensure_ascii=False),
                had,
                data["urgency"],
                json.dumps(data, ensure_ascii=False),
            ),
        )
        con.commit()
        con.close()
    except Exception:
        pass  # Do not fail request if local logging fails

    return {"disclaimer": DISCLAIMER, "data": data}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    analysis: dict | None = None
    patient: dict | None = None
    history: list[ChatMessage] = []


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not API_KEY:
        return JSONResponse({"error": "No Groq API key configured. Start the server with `python app.py` and enter it when prompted."}, 500)

    question = req.question.strip()
    if not question:
        return JSONResponse({"error": "Please enter a question."}, 400)

    # Build context from report analysis and patient details
    context_lines = []
    if req.patient:
        p = req.patient
        context_lines.append(f"Patient Name: {p.get('name', 'Patient')}, Sex: {p.get('sex', 'Not specified')}, DOB: {p.get('dob', 'Not specified')}")

    if req.analysis:
        a = req.analysis
        urgency = a.get("urgency", "")
        urgency_reason = a.get("urgency_reason", "")
        findings = a.get("findings", [])
        biomarkers = a.get("biomarkers", [])
        primary = a.get("primary", [])
        secondary = a.get("secondary", [])
        discrepancy = a.get("discrepancy", "")

        if urgency:
            context_lines.append(f"Overall Urgency Level: {urgency} - {urgency_reason}")
        if findings:
            context_lines.append("Key Findings from Report: " + "; ".join(str(f) for f in findings))
        if biomarkers:
            bm_strs = [
                f"{b.get('name')}: {b.get('value')} {b.get('unit')} (Range: {b.get('range')}, Status: {b.get('status')})"
                for b in biomarkers if isinstance(b, dict) and b.get("name")
            ]
            if bm_strs:
                context_lines.append("Biomarkers: " + " | ".join(bm_strs))
        if primary:
            prim_strs = [
                f"{p.get('name')}" + (f" [{p.get('probability')}]" if p.get('probability') else "") + f": {p.get('why')}"
                for p in primary if isinstance(p, dict) and p.get("name")
            ]
            if prim_strs:
                context_lines.append("Primary Considerations: " + " | ".join(prim_strs))
        if secondary:
            sec_strs = [
                f"{s.get('name')}" + (f" [{s.get('probability')}]" if s.get('probability') else "") + f": {s.get('why')}"
                for s in secondary if isinstance(s, dict) and s.get("name")
            ]
            if sec_strs:
                context_lines.append("Secondary Considerations: " + " | ".join(sec_strs))
        if discrepancy:
            context_lines.append(f"Noted Discrepancy: {discrepancy}")

    context_str = "\n".join(context_lines) if context_lines else "No previous report details provided."

    system_prompt = (
        "You are MedScan AI, an intelligent, empathetic medical assistant helping a patient understand their lab report, injuries, and symptoms.\n\n"
        f"PATIENT & REPORT CONTEXT:\n{context_str}\n\n"
        "GUIDELINES:\n"
        "- Never provide a diagnosis with certainty. Use terms like 'possible considerations include', 'findings suggest'.\n"
        "- When discussing possible conditions, differential diagnosis, or when asked about chances (e.g. 'what are the chances?', 'could it be X vs Y?'):\n"
        "  * Predict and rate the relative possibilities with estimated percentages (e.g., 65% fracture, 25% sprain, 10% contusion; or 60% viral infection, 25% drug reaction, etc.) based on their reported symptoms, trauma history, or lab values.\n"
        "  * Clearly explain the clinical reasoning behind why one condition is more or less likely than another.\n"
        "  * Describe the key differentiating signs that distinguish the top possibilities (e.g. pinpoint bone tenderness vs diffuse ligament pain).\n"
        "- Recommend specific confirmatory tests (e.g. 2-view X-ray, ultrasound, CT scan, CBC, cultures, antigen/antibody panels) that a doctor would order.\n"
        "- Provide safe immediate first aid and home care measures (e.g. RICE protocol: Rest, Ice, Compression, Elevation, splinting for trauma; hydration for fevers/illness; warnings against harmful actions or contraindicated medications like NSAIDs in bleeding risks).\n"
        "- Highlight critical emergency red flags that require going to the ER immediately (e.g. deformity, numbness, cold/pale extremities, uncontrolled bleeding, severe shortness of breath).\n"
        "- Structure your answer with clear markdown bullet points and bold section headers.\n"
        "- Keep responses concise (under 250 words) and easy to understand for patients."
    )

    messages = [{"role": "system", "content": system_prompt}]

    for msg in req.history[-6:]:
        if msg.role in ("user", "assistant") and msg.content:
            messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": question})

    try:
        import httpx
        import asyncio
        async with httpx.AsyncClient(timeout=60) as c:
            payload = {
                "model": MODEL,
                "temperature": 0.3,
                "max_tokens": 550,
                "messages": messages,
            }
            for attempt in range(2):
                r = await c.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload)
                if r.status_code == 429 and attempt == 0:
                    retry_after = min(float(r.headers.get("retry-after", 3.0)), 5.0)
                    await asyncio.sleep(retry_after)
                    continue
                r.raise_for_status()
                break
            ai_reply = r.json()["choices"][0]["message"]["content"]
            return {"reply": ai_reply}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return JSONResponse({"error": "The AI service is temporarily busy (rate limit). Please wait 15-30 seconds and try again."}, 429)
        return JSONResponse({"error": f"AI service error ({e.response.status_code}): {e.response.text[:150]}"}, 502)
    except Exception as e:
        return JSONResponse({"error": f"Could not reach AI service: {e.__class__.__name__}"}, 502)



@app.get("/")
def home():
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    return JSONResponse({"error": "index.html not found"}, 404)


if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main():
    global API_KEY, MODEL

    parser = argparse.ArgumentParser(description="Run the MedScan server.")
    parser.add_argument(
        "--api-key",
        "--groq-api-key",
        dest="api_key",
        required=True,
        help="Groq API key",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    API_KEY = args.api_key.strip()
    if not API_KEY:
        parser.error("the Groq API key cannot be empty")
    MODEL = args.model.strip()
    if not MODEL:
        parser.error("the model cannot be empty")

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
