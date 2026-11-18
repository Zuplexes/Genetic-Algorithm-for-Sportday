"""หาลิงก์ทุกสาขาจากการค้นหาใน Google Maps"""
import logging  # log จำนวนสาขาที่เจอ
import random  # สุ่มเวลารอ
import time  # time.sleep
from pathlib import Path  # สร้างโฟลเดอร์ของไฟล์ output
from urllib.parse import quote  # encode คำค้นภาษาไทยให้ใส่ใน URL ได้

import pandas as pd  # บันทึกผลเป็น CSV

from .browser import open_context  # เปิด Chrome พร้อม session
from .scraper import SEL, parse_place_url  # ใช้ selector และตัวแกะ URL ชุดเดียวกับ scraper

log = logging.getLogger(__name__)  # logger ชื่อ teenoi_data.branch_finder
COLUMNS = ["branch_id", "name", "lat", "lng", "gmaps_url", "found_by_query"]  # คอลัมน์ของ branches.csv
MAX_IDLE_SCROLLS = 5  # เลื่อนแล้วไม่เจอลิงก์ใหม่กี่รอบติดถือว่าสุดรายการ


def _collect_place_links(page):
    """เลื่อนรายการผลค้นหาจนสุด คืนชุด href ของทุกร้านที่เห็น"""
    hrefs, idle = {page.url}, 0  # ถ้าผลค้นหามีร้านเดียว Google จะพาเข้าหน้าร้านเลย จึงเก็บ URL ปัจจุบันไว้ด้วย
    links = page.locator(SEL["place_link"])  # ลิงก์ร้านทั้งหมดในหน้า (locator อ่านใหม่ทุกครั้งที่ใช้)
    while idle < MAX_IDLE_SCROLLS and links.count():  # หยุดเมื่อไม่มีอะไรใหม่หลายรอบ หรือไม่มีรายการเลย
        before = len(hrefs)  # จำนวนก่อนอ่านรอบนี้
        hrefs.update(links.evaluate_all("els => els.map(e => e.href)"))  # อ่าน href ทุกลิงก์ในครั้งเดียว
        idle = idle + 1 if len(hrefs) == before else 0  # ไม่มีลิงก์ใหม่ = นับรอบว่าง, มีใหม่ = เริ่มนับใหม่
        links.last.hover()  # เลื่อน panel ผลค้นหาโดยไม่ต้องพึ่ง class name ของ panel
        page.mouse.wheel(0, 4000)  # หมุนล้อเมาส์ลง ให้ Google โหลดผลเพิ่ม
        time.sleep(random.uniform(1.2, 2.5))  # รอผลชุดใหม่โหลด
    return hrefs


def find_branches(cfg):
    """ค้นทุก query ใน cfg["branch_finder"] คืน DataFrame คอลัมน์ COLUMNS (บันทึกลง output_file ถ้ากำหนด)"""
    bf = cfg["branch_finder"]  # ส่วน branch_finder: ใน config
    must = bf.get("name_must_contain") or ""  # ชื่อร้านต้องมีคำนี้ ("" = ไม่กรอง)
    found = {}  # branch_id -> row (dict กันสาขาซ้ำข้าม query)
    with open_context(cfg) as ctx:  # เปิด browser ครั้งเดียวใช้ทุก query
        page = ctx.pages[0] if ctx.pages else ctx.new_page()  # ใช้แท็บที่เปิดอยู่แล้วถ้ามี
        for query in bf["search_queries"]:  # ค้นทีละพื้นที่
            page.goto(f"https://www.google.com/maps/search/{quote(query)}?hl=th",  # หน้าผลค้นหา ภาษาไทย
                      wait_until="domcontentloaded", timeout=60000)  # รอแค่ HTML สูงสุด 60 วินาที
            try:
                page.wait_for_selector(SEL["place_link"], timeout=20000)  # รอรายการผลค้นหาขึ้น
            except Exception:  # ไม่ขึ้นรายการ (อาจเข้าหน้าร้านตรง หรือไม่เจอเลย)
                log.warning("ไม่พบผลค้นหาแบบรายการสำหรับ \"%s\"", query)
            n_new = 0  # สาขาใหม่จาก query นี้
            for href in _collect_place_links(page):  # ทุกลิงก์ที่เจอ
                p = parse_place_url(href)  # แกะ id ชื่อ พิกัด จากลิงก์
                if p["branch_id"] and must in (p["name"] or "") and p["branch_id"] not in found:  # มี id, ชื่อตรง, ยังไม่เคยเจอ
                    found[p["branch_id"]] = {**p, "gmaps_url": href, "found_by_query": query}  # เก็บพร้อมที่มา
                    n_new += 1
            log.info("\"%s\": เจอสาขาใหม่ %d สาขา (รวม %d)", query, n_new, len(found))
            time.sleep(random.uniform(2, 4))  # พักก่อนค้น query ถัดไป
    df = pd.DataFrame(list(found.values()), columns=COLUMNS)  # ตารางสาขา (ว่างก็มีคอลัมน์ครบ)
    if bf.get("output_file"):  # ตั้ง path ไว้ = บันทึกลงไฟล์
        Path(bf["output_file"]).parent.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี
        df.to_csv(bf["output_file"], index=False, encoding="utf-8")  # เขียนทับทุกรอบ
    return df
