"""ตรวจ logic ที่ไม่ต้องเปิด browser: python tests/test_offline.py"""
from datetime import datetime  # สร้างเวลาอ้างอิงตายตัว

import pandas as pd  # สร้างข้อมูลทดสอบ

from teenoi_data.cleaner import clean_dataframe, clean_text, detect_language, select_per_branch  # ฟังก์ชันที่ทดสอบ
from teenoi_data.scraper import approx_date, parse_place_url  # ฟังก์ชันที่ทดสอบ

# --- approx_date: ข้อความวันที่ -> วันที่โดยประมาณ ---
now = datetime(2026, 11, 15)  # เวลาอ้างอิงตายตัว ผลจะได้ไม่เปลี่ยนตามวันที่รัน
assert approx_date("3 เดือนที่แล้ว", now) == "2026-08-17"  # 3 x 30 วัน
assert approx_date("สัปดาห์ที่แล้ว", now) == "2026-11-08"  # ไม่มีตัวเลข = 1 สัปดาห์
assert approx_date("แก้ไขเมื่อ 2 ปีที่แล้ว", now) == "2024-11-15"  # มีข้อความนำหน้าก็อ่านได้
assert approx_date("5 ชั่วโมงที่ผ่านมา", now) == "2026-11-15"  # ไม่ถึงวัน = วันนี้
assert approx_date("") is None  # อ่านไม่ออก = None

# --- parse_place_url: แกะชื่อ id พิกัด จากลิงก์ ---
p = parse_place_url(
    "https://www.google.com/maps/place/%E0%B8%AA%E0%B8%B8%E0%B8%81%E0%B8%B5%E0%B9%89+MBK/"  # ชื่อ "สุกี้ MBK" แบบ encode
    "data=!4m7!3m6!1s0x30e29fad0d0c9703:0xf3b92c60a2ac3177!8m2!3d13.7443!4d100.53!16s%2Fg%2F1?hl=th")  # id + lat/lng
assert p == {"name": "สุกี้ MBK", "branch_id": "0x30e29fad0d0c9703:0xf3b92c60a2ac3177", "lat": 13.7443, "lng": 100.53}
assert parse_place_url("https://maps.app.goo.gl/abc")["branch_id"] is None  # ลิงก์สั้นยังไม่มี id

# --- clean_text ---
assert clean_text("(แปลโดย Google) Good food\n\n(ต้นฉบับ)\nอาหารดี") == "อาหารดี"  # เก็บแค่ต้นฉบับ
assert clean_text("อร่อยมากกกกกก   ดู https://x.co/a <b>ดี</b> …") == "อร่อยมากก ดู ดี"  # ลดตัวซ้ำ ลบ URL/HTML/…
assert clean_text("ไม่อร่อย ราคา 1000 บาท") == "ไม่อร่อย ราคา 1000 บาท"  # "ไม่" และตัวเลขต้องอยู่ครบ
assert clean_text("ดี 👍", {"keep_emoji": False}) == "ดี"  # ลบ emoji เมื่อสั่ง
assert clean_text(None) == "" and clean_text(float("nan")) == ""  # ค่าว่างได้ ""

# --- detect_language ---
assert [detect_language(t) for t in ["อร่อย good", "Good food", "美味しい", "👍👍👍"]] == ["th", "en", "other", "other"]

# --- clean_dataframe: กรองแถวและใส่ review_id ---
df = clean_dataframe(pd.DataFrame({"review": ["อร่อยมาก", "อร่อยมาก", "ดี", None, "美味しいです", "Great value"], "rating": [5, 5, 4, 3, 5, 5]}))
assert df["review"].tolist() == ["อร่อยมาก", "Great value"] and df["review_id"].tolist() == [1, 2]  # เหลือ 2 แถว id 1, 2
assert list(df.attrs["dropped"].values()) == [2, 1, 1]  # สั้น 2 (ดี, None), ภาษา 1, ซ้ำ 1

# --- select_per_branch ---
# คัดสาขาละ 3: ดาว <= 4 ก่อน (ใหม่สุดก่อน) แล้วเติม 5 ดาว; รอบดึงหลัง (scraped_at มากกว่า) ถือว่าใหม่กว่า
sel = select_per_branch(pd.DataFrame({
    "review_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "branch_id": ["a", "a", "a", "a", "a", "b", "b", "a"],
    "rating":    [5, 2, 5, 4, 1, 5, 3, 5],
    "scraped_at": ["t1"] * 7 + ["t2"],
}), 3)
assert sel["review_id"].tolist() == [2, 4, 5, 6, 7]  # a: ดาวต่ำ 3 แถวเต็มโควตา, b: มีแค่ 2 แถว
sel = select_per_branch(pd.DataFrame({
    "review_id": [1, 2, 3, 4], "branch_id": ["a"] * 4, "rating": [5, 2, 5, 5], "scraped_at": ["t1", "t1", "t1", "t2"],
}), 3)
assert sel["review_id"].tolist() == [1, 2, 4]  # เติม 5 ดาว: แถว 4 (รอบใหม่) ก่อน แล้วแถว 1
sel = select_per_branch(pd.DataFrame({
    "review_id": [1, 2, 3, 4, 5, 6, 7], "branch_id": ["a"] * 7, "rating": [5, 5, 5, 5, 4, 4, 1], "scraped_at": ["t1"] * 7,
}), 5, priority_max_rating=3)
assert sel["review_id"].tolist() == [1, 2, 5, 6, 7]  # ดาวต่ำ 1 แถว + เติม 4 กับ 5 ดาวอย่างละ 2 (ไม่ใช่ 5 ดาวล้วน)

print("ok")  # ถึงบรรทัดนี้ = ผ่านทุก assert
