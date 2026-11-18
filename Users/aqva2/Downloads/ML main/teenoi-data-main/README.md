# teenoi-data

ดึงรีวิวร้านสุกี้ตี๋น้อยจาก Google Maps แล้ว clean ให้พร้อม import เข้า Label Studio ในคำสั่งเดียว
และเป็น package (`teenoi_data`) ให้ `teenoi-predict` ใช้ scraper/cleaner ตัวเดียวกัน

## ติดตั้ง

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

ต้องมี Google Chrome ในเครื่อง (ใช้ทั้งตอนล็อกอินและตอนดึงรีวิว จึงไม่ต้อง `playwright install chromium`)

repo นี้มี `configs/Data_config.yml` มาด้วยแล้ว แต่ไม่มี `data/` ต้องรัน `python main.py` เพื่อดึงรีวิวเอง

## วิธีรัน

```bash
python login.py     # ครั้งแรกครั้งเดียว: ล็อกอิน Google เองในหน้าต่าง Chrome ที่เปิดขึ้น แล้วปิดหน้าต่าง (ต้องมี Google Chrome ในเครื่อง)
python main.py      # ได้ data/clean/reviews_clean.csv
```

ไม่มี command-line arguments ทุกอย่างตั้งใน `configs/Data_config.yml`:

- ใส่ลิงก์สาขาใน `scraper.urls` (ลิงก์สั้น `maps.app.goo.gl` หรือลิงก์เต็มก็ได้) หรือตั้ง `steps.find_branches: true` ให้ค้นหาสาขาเอง
- เปิด/ปิดแต่ละขั้นที่ `steps:`
- `browser.channel` ต้องเป็น `chrome` เสมอ เพราะ Google ไม่ยอมให้ล็อกอินใน Chromium ของ Playwright
- ถ้า log เตือน "มุมมองแบบจำกัด" แปลว่า session หลุด ให้รัน `python login.py` ใหม่

รันซ้ำได้: รีวิวที่อยู่ใน `reviews_raw.csv` แล้วจะไม่ถูกดึงซ้ำ (ดึงเฉพาะรีวิวที่ใหม่กว่า) และรีวิวเดิมได้ `review_id` เดิม
ถ้าต้องการดึงใหม่ทั้งหมดให้ลบ `data/raw/reviews_raw.csv`

## ไฟล์ output

| ไฟล์ | คอลัมน์ |
|---|---|
| `data/raw/branches.csv` (เมื่อ `find_branches: true`) | `branch_id, name, lat, lng, gmaps_url, found_by_query` |
| `data/raw/reviews_raw.csv` | `branch_id, branch, rating, date_text, date_approx, review, gmaps_review_id, source_url, scraped_at` |
| `data/clean/reviews_clean.csv` → import เข้า Label Studio | `review_id, review` |
| `data/clean/reviews_clean_full.csv` | `review_id, review` + คอลัมน์ของ raw |
| `data/raw/<ชื่อสาขา>.csv` | เหมือน `reviews_raw.csv` แต่เฉพาะสาขานั้น |
| `data/clean/<ชื่อสาขา>.csv` → import เข้า Label Studio ทีละสาขาได้ | `review_id, review` (เลขเดียวกับไฟล์รวม ไม่ซ้ำกันข้ามสาขา) |
| `logs/run_*.log` | log ของแต่ละรอบ |

ไม่เก็บชื่อผู้รีวิว

## Label Studio

สร้างโปรเจกต์ → Labeling Interface → Code → วางเนื้อหาจาก `labelstudio_config.xml` → import `data/clean/reviews_clean.csv`
แนะนำ export ผล label เป็น **JSON** (เชื่อม span กับ sentiment ได้ครบ)

## ขั้นตอน clean

1. ตัดส่วนที่ Google แปลอัตโนมัติ เก็บเฉพาะข้อความต้นฉบับ
2. ลบ URL และ HTML tag
3. `pythainlp.util.normalize`
4. ลดตัวอักษรซ้ำ (ไม่แตะตัวเลข)
5. ตัด "…" ท้ายข้อความ และลดช่องว่างซ้ำ
6. ตัดแถวที่สั้นเกินไป / ภาษาไม่ใช่ไทย-อังกฤษ / ข้อความซ้ำ

ไม่ตัดคำและไม่ลบ stopwords (จะทำให้ "ไม่อร่อย" กลายเป็น "อร่อย")

## ใช้เป็น package

```python
from teenoi_data import __version__
from teenoi_data.cleaner import clean_text, DEFAULT_CLEANING
from teenoi_data.scraper import scrape_many, scrape_place
from teenoi_data.branch_finder import find_branches
```

ทุกครั้งที่แก้ cleaner หรือ `SEL` ใน `scraper.py` ต้องเพิ่ม `__version__` และสร้าง git tag ใหม่

## ทดสอบ

```bash
python tests/test_offline.py
python tests/test_split.py
```

## ข้อควรระวัง

- การ scrape ขัดกับ Terms of Service ของ Google ควรใช้บัญชี Google สำรองและดึงในปริมาณพอเหมาะ
- `data/` มีข้อความรีวิวที่ดึงมาจาก Google Maps จึงไม่ commit ขึ้น repo
- `chrome_profile/` (ตาม `browser.profile_dir`) มี session ที่ล็อกอินอยู่ ห้าม commit หรือแชร์
- Google เปลี่ยน class name บ่อย ถ้าดึงได้ 0 รีวิวให้แก้ dict `SEL` ใน `src/teenoi_data/scraper.py`
