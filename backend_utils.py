import os
import smtplib
import random
import string
from email.mime.text import MIMEText
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI") 
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

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
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        return False, "Server Error: Email Not Configured."

    msg = MIMEText(f"Your Verification Code is: {code}")
    msg['Subject'] = "Your Verification Code"
    msg['From'] = EMAIL_SENDER
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, to_email, msg.as_string())
        return True, "Code sent!"
    except Exception as e:
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