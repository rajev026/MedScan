# MedScan setup (FastAPI + Groq Multimodal AI)

1. Get a Groq API key at https://console.groq.com/keys.
2. Open a terminal in this folder.
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`

4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Start the server:
   $env:GROQ_API_KEY = " Enter your groq API key here "
   $env:MODEL = "qwen/qwen3.8-27b"

   python -m uvicorn app:app --reload
   ```
   Note: passing a key as a command-line argument may leave it in your terminal history or visible to other local processes. Use a trusted machine and avoid sharing terminal logs.

6. Open your browser at http://localhost:8000
