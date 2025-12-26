import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import app as main_app
import backend_utils as backend
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# --- 1. APIs ---
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


app = gr.mount_gradio_app(app, main_app.demo, path="/gradio")

app.mount("/", StaticFiles(directory="static", html=True), name="static")