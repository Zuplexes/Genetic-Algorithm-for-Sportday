"""โหลด config (YAML) + ตั้งค่า logging"""
import logging  # ระบบ log มาตรฐานของ Python
import sys  # ใช้ sys.stdout สำหรับพิมพ์ log ออกหน้าจอ
from datetime import datetime  # ใช้ตั้งชื่อไฟล์ log ตามเวลาที่รัน
from pathlib import Path  # จัดการ path แบบข้าม OS

import yaml  # อ่านไฟล์ .yml (package pyyaml)


def _get(cfg, dotted):
    """อ่านค่าจาก dict ซ้อนด้วย key แบบจุด เช่น "scraper.output_file" ไม่เจอคืน None"""
    for key in dotted.split("."):  # ไล่ทีละชั้น: "scraper" แล้ว "output_file"
        if not isinstance(cfg, dict) or key not in cfg:  # ชั้นนี้ไม่ใช่ dict หรือไม่มี key นี้
            return None  # ถือว่าไม่มีค่า
        cfg = cfg[key]  # ลงไปอีกหนึ่งชั้น
    return cfg  # ค่าของ key ชั้นในสุด


def load_config(path, required=()):
    """อ่านไฟล์ YAML แล้วตรวจว่ามี key ที่จำเป็นครบ (เขียน key ซ้อนด้วยจุด เช่น "scraper.output_file")"""
    path = Path(path)  # รับได้ทั้ง str และ Path
    if not path.exists():  # ไฟล์ config หาย (ปกติมาพร้อม repo)
        raise SystemExit(f"ไม่พบไฟล์ config: {path}")  # จบโปรแกรมพร้อมบอกว่าไฟล์ไหนหาย
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}  # safe_load ไม่รันโค้ดใน YAML, ไฟล์ว่างได้ {}
    missing = [k for k in required if _get(cfg, k) is None]  # key ที่จำเป็นแต่ไม่มีค่า
    if missing:  # ขาดอย่างน้อยหนึ่ง key
        raise SystemExit(f"ไฟล์ config {path} ขาด key ที่จำเป็น: {', '.join(missing)}")  # หยุดก่อนเริ่มงานจริง
    return cfg  # dict ของ config ทั้งไฟล์


def setup_logging(level="INFO", log_dir="logs", name="run"):
    """log ลงทั้งหน้าจอและไฟล์ใน log_dir คืน path ของไฟล์ log"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ logs ถ้ายังไม่มี
    log_file = Path(log_dir) / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"  # เช่น logs/run_20261115_092526.log
    if hasattr(sys.stdout, "reconfigure"):  # คอนโซล Windows พิมพ์ภาษาไทยไม่ได้ถ้าไม่ใช่ utf-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ตัวที่พิมพ์ไม่ได้แทนด้วย ? แทนการ crash
    logging.basicConfig(  # ตั้งค่า root logger ครั้งเดียว ทุก module ใช้ร่วมกัน
        level=getattr(logging, str(level).upper(), logging.INFO),  # "info" -> logging.INFO, สะกดผิดใช้ INFO
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",  # เวลา ระดับ ชื่อ module: ข้อความ
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],  # จอ + ไฟล์
        force=True,  # ล้าง handler เดิม (กันกรณีถูกเรียกซ้ำหรือ library อื่นตั้งไว้ก่อน)
    )
    return log_file  # ให้ main.py พิมพ์ path ไฟล์ log ในสรุปท้าย
