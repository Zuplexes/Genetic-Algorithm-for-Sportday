"""เปิด persistent browser (เก็บ session Google ไว้ใน profile_dir) + ล็อกอินด้วยมือ"""
import shutil  # shutil.which หา chrome.exe ใน PATH
import subprocess  # เปิด Chrome จริงเป็นโปรเซสแยก
from contextlib import contextmanager  # ทำให้ open_context ใช้กับ with ได้
from pathlib import Path  # จัดการ path ของ profile

from playwright.sync_api import sync_playwright  # ควบคุม browser แบบ synchronous

# ต้องเป็น UA ของ Chrome desktop ถ้าใช้ UA ของ headless Chromium จะไม่มีแท็บ "รีวิว"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "  # ส่วนระบุ OS + engine
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"  # ส่วนระบุเวอร์ชัน Chrome
)
MAPS_URL = "https://www.google.com/maps?hl=th"  # หน้าแรกที่เปิดตอนล็อกอิน (hl=th = ภาษาไทย)
DEFAULT_PROFILE = "gmaps_profile"  # โฟลเดอร์ profile ถ้า config ไม่ได้ระบุ browser.profile_dir


def _profile_dir(cfg):
    """path เต็มของโฟลเดอร์ profile ที่เก็บ session Google"""
    return Path(cfg.get("browser", {}).get("profile_dir", DEFAULT_PROFILE)).resolve()  # resolve ให้เป็น path เต็ม


@contextmanager
def open_context(cfg, headless=None):
    """with open_context(cfg) as ctx: page = ctx.new_page()"""
    b = cfg.get("browser", {})  # ส่วน browser: ใน config
    with sync_playwright() as p:  # เริ่ม Playwright (ปิดให้เองเมื่อออกจาก with)
        ctx = p.chromium.launch_persistent_context(  # persistent = ใช้ profile เดิม จึงยังล็อกอินอยู่
            user_data_dir=str(_profile_dir(cfg)),  # โฟลเดอร์ session ที่ได้จาก login.py
            headless=b.get("headless", True) if headless is None else headless,  # argument ชนะค่าใน config
            channel=b.get("channel") or None,  # "chrome" = ใช้ Google Chrome ในเครื่องแทน Chromium
            locale=b.get("locale", "th-TH"),  # ให้หน้าเว็บเป็นภาษาไทย (ข้อความปุ่มใน TXT เป็นภาษาไทย)
            viewport=b.get("viewport") or {"width": 1280, "height": 900},  # ขนาดหน้าต่าง
            user_agent=USER_AGENT,  # ปลอมเป็น Chrome desktop ปกติ
            args=["--blink-settings=imagesEnabled=false"],  # ไม่โหลดรูป หน้าเบาและเลื่อนไวขึ้น (ดาว/ข้อความไม่ใช่รูป)
        )
        try:
            yield ctx  # ส่ง context ให้โค้ดใน with ใช้งาน
        finally:
            ctx.close()  # ปิด browser เสมอ แม้เกิด error กลางทาง


def login(cfg):
    """เปิดหน้าต่างให้ผู้ใช้ล็อกอิน Google เอง (ไม่กรอก email/password อัตโนมัติ)"""
    # ponytail: เปิด Chrome จริงตรง ๆ ไม่ผ่าน Playwright เพราะ Google ไม่ให้ล็อกอินใน browser ที่ถูก automation คุม
    # ("เบราว์เซอร์หรือแอปนี้อาจไม่ปลอดภัย") ต้องใช้คู่กับ browser.channel: chrome ตอน scrape
    chrome = shutil.which("chrome") or r"C:\Program Files\Google\Chrome\Application\chrome.exe"  # หาใน PATH ก่อน ไม่เจอใช้ path มาตรฐาน
    print("ล็อกอินบัญชี Google ในหน้าต่าง Chrome ที่เปิดขึ้นมา แล้วปิดหน้าต่างเพื่อบันทึก session")  # บอกผู้ใช้ว่าต้องทำอะไร
    subprocess.run([chrome, f"--user-data-dir={_profile_dir(cfg)}", "--no-first-run", MAPS_URL])  # รอจนผู้ใช้ปิดหน้าต่าง
