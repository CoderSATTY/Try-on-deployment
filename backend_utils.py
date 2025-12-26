import os
import random
import string
import resend
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI") 
RESEND_API_KEY = os.getenv("RESEND_API_KEY")

db = None
users_collection = None

if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI)
        db = client['tryon_app']
        users_collection = db['users']
    except Exception:
        pass

def generate_code():
    return ''.join(random.choices(string.digits, k=6))

def send_email(to_email, code):
    if not RESEND_API_KEY:
        return False, "Server Error: Email Not Configured."

    resend.api_key = RESEND_API_KEY

    try:
        # Uses HTTP (Port 443) which Render allows
        r = resend.Emails.send({
            "from": "onboarding@resend.dev",  # Use this default for testing, or your verified domain
            "to": to_email,
            "subject": "Your Verification Code",
            "html": f"<p>Your Verification Code is: <strong>{code}</strong></p>"
        })
        return True, "Code sent!"
    except Exception as e:
        print(f"Email Error: {e}") 
        return False, f"Email Failed: {str(e)}"

def register_user(email, name):
    if users_collection is None: 
        return False, "Database Error: Not Connected"
    
    code = generate_code()
    
    email_success, email_msg = send_email(email, code)
    if not email_success:
        return False, email_msg 

    try:
        users_collection.update_one(
            {"email": email},
            {
                "$set": {
                    "verification_code": code, 
                    "last_login": datetime.utcnow(),
                    "name": name
                },
                "$setOnInsert": {"generation_count": 0}
            },
            upsert=True
        )
        return True, "Code sent!"
    except Exception:
        return False, "Database Write Error"

def verify_code(email, code):
    if users_collection is None: return False, "Database Error"
    
    user = users_collection.find_one({"email": email})
    if not user: return False, "User not found."
    
    if str(user.get("verification_code")).strip() == str(code).strip():
        return True, "Verified!"
    return False, "Invalid code."

def check_quota(email):
    if users_collection is None: return True, 3 
    
    user = users_collection.find_one({"email": email})
    if not user: return True, 3 
    
    count = user.get("generation_count", 0)
    if count < 3:
        return True, 3 - count
    return False, 0

def increment_usage(email):
    if users_collection is not None:
        users_collection.update_one(
            {"email": email},
            {"$inc": {"generation_count": 1}}
        )