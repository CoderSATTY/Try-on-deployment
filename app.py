import gradio as gr
import requests 
import io
import cloudinary
import cloudinary.uploader
from PIL import Image
from dotenv import load_dotenv
import os
import asyncio
import backend_utils as backend

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

API_URL = "https://me240003014--tryon-inference-fastapi-app.modal.run/generate" 

def get_user_from_url(request: gr.Request):
    if request:
        return request.query_params.get('user')
    return None

async def process_tryon(email, subject_file, subject_url_input, garment_file, garment_url_input):
    # 1. Authentication Check
    if not email:
        raise gr.Error("Authentication failed. Please login again.")

    # 2. Quota Check (Strict 3 tries limit)
    allowed, remaining = backend.check_quota(email)
    if not allowed:
        raise gr.Error("Quota exceeded. You have used all 3 tries.")

    # 3. File Processing
    def get_url_and_id_sync(file_obj, url_str):
        if url_str and url_str.strip():
            return url_str.strip(), None
        elif file_obj is not None:
            response = cloudinary.uploader.upload(file_obj)
            return response["secure_url"], response["public_id"]
        return None, None

    sub_url, sub_id = await asyncio.to_thread(get_url_and_id_sync, subject_file, subject_url_input)
    garm_url, garm_id = await asyncio.to_thread(get_url_and_id_sync, garment_file, garment_url_input)

    if not sub_url or not garm_url:
        raise gr.Error("Please provide both Subject and Garment images.")

    try:
        payload = {
            "subject_url": sub_url,
            "garment_url": garm_url
        }

        # 4. API Call
        response = await asyncio.to_thread(requests.post, API_URL, json=payload, timeout=600)
        
        if response.status_code == 200:
            image_bytes = response.content
            
            # 5. Increment Usage on Success
            backend.increment_usage(email)
            
            # --- UPDATE STATUS HEADER ---
            new_remaining = max(0, remaining - 1)
            status_msg = f"<div class='logged-in-message'>Logged in as: {email} | Credits: {new_remaining}/3</div>"
            
            # Return Image AND Updated Status Message
            return [Image.open(io.BytesIO(image_bytes))], status_msg
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            raise gr.Error(f"Generation failed: {response.status_code}")
            
    except Exception as e:
        print(f"Connection Error: {e}")
        raise gr.Error(f"Connection Error: {str(e)}")

    finally:
        if sub_id:
            await asyncio.to_thread(cloudinary.uploader.destroy, sub_id)
        if garm_id:
            await asyncio.to_thread(cloudinary.uploader.destroy, garm_id)

custom_css = """
/* --- Layout CSS --- */
.output-gallery-class {
    height: 75vh !important; 
    min-height: 500px !important;
    display: flex;
    flex-direction: column;
}

.output-gallery-class .grid-wrap, 
.output-gallery-class .grid-container {
    height: 100% !important;
}

.output-gallery-class img {
    height: 100% !important;
    width: 100% !important;
    object-fit: contain !important;
    object-position: center !important;
    display: block;
}

.output-gallery-class .thumbnails {
    display: none !important;
}

.input-image-class {
    max-height: 300px !important;
}

.input-image-class img {
    object-fit: contain !important; 
    max-height: 280px !important;
}

.or-divider {
    text-align: center;
    font-weight: bold;
    color: #ffffff; /* White Divider */
    margin: 5px 0;
}

/* --- Color Scheme: Black, Orange, White --- */
body, .gradio-container {
    background-color: #000000; /* Pure Black Background */
    color: #ffffff; /* White Text */
}

/* Headings */
h1, h2, h3, h4, span, p {
    color: #ffffff !important;
}

/* Buttons (Force Orange) */
button.primary {
    background-color: #ff6600 !important; /* Orange */
    color: #ffffff !important;
    border: none !important;
}

/* Specific styling for the Logged In message */
.logged-in-message {
    font-size: 16px !important;
    color: #ff6600 !important; /* Orange Text */
    font-weight: bold;
    border: 1px solid #ff6600; /* Orange Border */
    padding: 10px;
    border-radius: 8px;
    background-color: rgba(255, 102, 0, 0.1); /* Transparent Orange Background */
    text-align: center;
    margin-bottom: 20px;
}
"""


with gr.Blocks(title="Virtual Try-On", css=custom_css, theme=gr.themes.Base(primary_hue="orange", neutral_hue="slate")) as demo:
    
    user_email_state = gr.State()
    
    # Logic to load user and format the message as HTML
    def load_user_data(request: gr.Request):
        email = get_user_from_url(request)
        if not email:
            # Red/Orange warning styled message if no user
            return None, "<div class='logged-in-message' style='color: #ff4500 !important; border-color: #ff4500;'>⚠️ No user detected. Please log in via the main URL.</div>"
        
        _, remaining = backend.check_quota(email)
        # Standard Orange styled message
        return email, f"<div class='logged-in-message'>Logged in as: {email} | Credits: {remaining}/3</div>"

    status_header = gr.HTML()

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Subject Image")
                subject_image = gr.Image(label="Upload Subject", type="filepath", elem_classes="input-image-class")
                gr.HTML("<div class='or-divider'>— OR —</div>")
                subject_url = gr.Textbox(label="Paste Subject URL", placeholder="https://...")

            with gr.Group():
                gr.Markdown("### Garment Image")
                garment_image = gr.Image(label="Upload Garment", type="filepath", elem_classes="input-image-class")
                gr.HTML("<div class='or-divider'>— OR —</div>")
                garment_url = gr.Textbox(label="Paste Garment URL", placeholder="https://...")

        with gr.Column(scale=1):
            output_gallery = gr.Gallery(
                label="Result", 
                columns=1, 
                rows=1,
                show_label=True,
                elem_classes="output-gallery-class",
                preview=True, 
                interactive=False
            )
            run_button = gr.Button("Run Try-On", variant="primary", size="lg")

    demo.load(load_user_data, inputs=None, outputs=[user_email_state, status_header])
    
    demo.queue(default_concurrency_limit=None)
    
    run_button.click(
        fn=process_tryon,
        inputs=[user_email_state, subject_image, subject_url, garment_image, garment_url],
        outputs=[output_gallery, status_header]  # Added status_header to outputs
    )

# app.py
if __name__ == "__main__":
    # Render provides the PORT env variable. Default to 7860 locally.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)