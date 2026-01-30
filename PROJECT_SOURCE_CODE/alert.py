import os
import smtplib
from email.message import EmailMessage
from playsound import playsound

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_WAV_PATH = os.path.join(BASE_DIR, "Alarm.wav")

EMAIL_ADDRESS = os.getenv("ALERT_EMAIL", "myktybekabdykaiymov@gmail.com")
EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASS", "xnmn xhnx vvgn rfko")
TO_EMAIL = os.getenv("ALERT_TO_EMAIL", "kafuteekito@gmail.com")


def send_email_alert(image_path: str):
    """Send email when unknown face is detected."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD or not TO_EMAIL:
        print("[EMAIL] Missing env vars ALERT_EMAIL / ALERT_EMAIL_PASS / ALERT_TO_EMAIL")
        return

    msg = EmailMessage()
    msg["Subject"] = "Alert! Unknown face detected"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.set_content("An unknown person was detected. See attached image.")

    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
        filename = os.path.basename(image_path)
        msg.add_attachment(img_data, maintype="image", subtype="png", filename=filename)
    except Exception as e:
        print(f"[EMAIL] Could not attach image: {e}")
        return

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"[EMAIL SENT] {image_path}")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")


def play_alarm():
    try:
        playsound(ALARM_WAV_PATH)
    except Exception as e:
        print(f"[ALARM ERROR] {e}")
