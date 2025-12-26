import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import app as main_app  # This imports the demo from app.py
import backend_utils as backend
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# --- 1. APIs (For your Login Page) ---
class LoginRequest(BaseModel):
    email: str
    name: str

class VerifyRequest(BaseModel):
    email: str
    code: str

@app.post("/api/login")
async def api_login(req: LoginRequest):
    success, msg = backend.register_user(req.email, req.name)
    return {"success": success, "message": msg}

@app.post("/api/verify")
async def api_verify(req: VerifyRequest):
    success, msg = backend.verify_code(req.email, req.code)
    return {"success": success, "message": msg}

# --- 2. Mount Gradio App ---
# IMPORTANT: Mount this BEFORE the static files so /gradio isn't blocked
# Users will access the app at: http://localhost:7860/gradio?user=...
app = gr.mount_gradio_app(app, main_app.demo, path="/gradio")

# --- 3. Mount Static Files (Frontend) ---
# This serves your index.html (Login Page) at the root URL "/"
# Ensure you have a folder named 'static' with index.html inside it
app.mount("/", StaticFiles(directory="static", html=True), name="static")