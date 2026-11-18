"""ตรวจการแยกไฟล์ต่อสาขา: python tests/test_split.py"""
import sys  # แก้ sys.path
import tempfile  # โฟลเดอร์ชั่วคราว ลบเองเมื่อจบ
from pathlib import Path  # จัดการ path

import pandas as pd  # สร้าง/อ่านข้อมูลทดสอบ

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ให้ import main.py จาก root ของ repo ได้
from main import split_by_branch  # ฟังก์ชันที่ทดสอบ

df = pd.DataFrame({"review_id": [1, 2, 3], "review": ["ก", "ข", "ค"], "branch": ["ตี๋น้อย A/B", "ตี๋น้อย C", "ตี๋น้อย A/B"]})  # "/" ห้ามใช้ในชื่อไฟล์
with tempfile.TemporaryDirectory() as d:  # เขียนลงโฟลเดอร์ชั่วคราว ไม่แตะ data/
    assert split_by_branch(df, d, ["review_id", "review"]) == 2  # 2 สาขา = 2 ไฟล์
    assert sorted(p.name for p in Path(d).iterdir()) == ["ตี๋น้อย A B.csv", "ตี๋น้อย C.csv"]  # "/" ถูกแทนด้วยช่องว่าง
    a = pd.read_csv(Path(d) / "ตี๋น้อย A B.csv")  # อ่านไฟล์สาขา A/B กลับมา
    assert a["review_id"].tolist() == [1, 3] and list(a.columns) == ["review_id", "review"]  # แถวและคอลัมน์ถูกต้อง
print("ok")  # ถึงบรรทัดนี้ = ผ่านทุก assert
