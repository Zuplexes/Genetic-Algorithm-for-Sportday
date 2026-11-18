"""รันครั้งแรกครั้งเดียว: เปิด browser ให้ล็อกอิน Google เอง session เก็บใน browser.profile_dir"""
import os  # os.chdir
from pathlib import Path  # หาโฟลเดอร์ของไฟล์นี้

os.chdir(Path(__file__).resolve().parent)  # path ใน config เป็น relative จาก root ของ repo

from teenoi_data.browser import login  # เปิด Chrome ให้ล็อกอินเอง
from teenoi_data.config import load_config  # อ่าน config

if __name__ == "__main__":  # รันด้วย python login.py
    login(load_config("configs/Data_config.yml", ["browser.profile_dir"]))  # ต้องมี profile_dir จึงรู้ว่าจะเก็บ session ที่ไหน
