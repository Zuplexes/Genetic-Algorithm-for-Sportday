"""ดึงรีวิว + clean ทั้งหมด: python main.py (ตั้งค่าทุกอย่างใน configs/Data_config.yml)"""
import logging  # log ของ main
import os  # os.chdir
import random  # ตั้ง seed
import re  # ลบอักขระต้องห้ามในชื่อไฟล์
from pathlib import Path  # จัดการ path

os.chdir(Path(__file__).resolve().parent)  # path ใน config เป็น relative จาก root ของ repo จึงย้ายมาที่นี่ก่อน

import pandas as pd  # อ่าน/เขียน CSV  (import หลัง chdir ตั้งใจ)

from teenoi_data.config import load_config, setup_logging  # โหลด config + ตั้ง log

CONFIG = "configs/Data_config.yml"  # ไฟล์ config หลัก
REQUIRED = [  # key ที่ต้องมีใน config ไม่งั้นหยุดตั้งแต่เริ่ม
    "steps", "browser.profile_dir", "scraper.output_file", "scraper.max_reviews_per_place",
    "cleaning.input_file", "cleaning.output.file", "cleaning.output.full_file",
]
log = logging.getLogger("main")  # logger ชื่อ main


def read_urls(scraper_cfg):
    """รวมลิงก์จาก scraper.urls และ scraper.urls_file ตัดซ้ำและตัดลิงก์ตัวอย่าง"""
    urls = list(scraper_cfg.get("urls") or [])  # ลิงก์ใน config (None -> [])
    if scraper_cfg.get("urls_file"):  # มีไฟล์ลิงก์เพิ่ม
        urls += Path(scraper_cfg["urls_file"]).read_text(encoding="utf-8").split()  # แยกด้วยช่องว่าง/ขึ้นบรรทัด
    urls = [u for u in dict.fromkeys(urls) if "xxxxxxxx" not in u]  # dict.fromkeys ตัดซ้ำโดยคงลำดับ, ตัดลิงก์ตัวอย่าง
    if not urls:  # ไม่มีลิงก์จริงเลย
        raise SystemExit(f"ยังไม่มีลิงก์สาขาให้ดึง: ใส่ลิงก์ใน scraper.urls ของ {CONFIG} หรือตั้ง steps.find_branches: true")
    return urls


def split_by_branch(df, out_dir, columns=None):
    """เขียน <ชื่อสาขา>.csv ต่อสาขาลง out_dir (ตัดมาจากไฟล์รวม เขียนทับทุกรอบ) คืนจำนวนไฟล์"""
    # ponytail: แยกด้วยชื่อสาขา ถ้ามีสองสาขาชื่อเหมือนกันเป๊ะจะรวมเป็นไฟล์เดียว ถ้าเจอค่อยต่อ branch_id ท้ายชื่อ
    n = 0  # จำนวนไฟล์ที่เขียน
    for name, group in df.groupby("branch"):  # ทีละสาขา
        safe = re.sub(r'[\\/:*?"<>|\s]+', " ", str(name)).strip(" .")  # ตัดอักขระที่ Windows ห้ามใช้ในชื่อไฟล์
        group[columns or list(df.columns)].to_csv(Path(out_dir) / f"{safe}.csv", index=False, encoding="utf-8")  # เลือกคอลัมน์แล้วเขียน
        n += 1
    return n


def run_find_branches(cfg, summary, outputs):
    """ขั้น find_branches: ค้นหาสาขา คืน DataFrame ของสาขา"""
    from teenoi_data.branch_finder import find_branches  # import เมื่อใช้ (ต้องมี playwright)
    if not (cfg.get("branch_finder") or {}).get("search_queries"):  # ไม่มีคำค้น
        raise SystemExit(f"ไฟล์ config {CONFIG} ขาด key ที่จำเป็น: branch_finder.search_queries")
    branches = find_branches(cfg)  # เปิด browser ค้นทุก query
    summary["สาขาที่ค้นเจอ"] = len(branches)  # ใส่ในสรุปท้าย
    outputs.append(cfg["branch_finder"].get("output_file"))  # path ของ branches.csv (None ถ้าไม่ได้ตั้ง)
    return branches


def run_scrape(cfg, urls, raw_path, summary, outputs):
    """ขั้น scrape: ดึงรีวิวต่อท้ายไฟล์ raw (ไม่ดึงรีวิวที่มีอยู่แล้ว) แล้วแยกไฟล์ต่อสาขา"""
    from teenoi_data.scraper import RAW_COLUMNS, scrape_many  # import เมื่อใช้ (ต้องมี playwright)
    # รีวิวที่เคยดึงแล้วไม่ดึงซ้ำ รันซ้ำได้โดยไฟล์ raw ไม่มีแถวซ้ำ (อยากดึงใหม่ทั้งหมดให้ลบไฟล์ raw)
    known = {}  # branch_id -> set(gmaps_review_id)
    if raw_path.exists():  # เคยดึงมาก่อน
        old = pd.read_csv(raw_path, dtype=str)  # อ่านเป็น str ทั้งหมด id จะได้ตรงกับที่อ่านจากหน้าเว็บ
        known = old.groupby("branch_id")["gmaps_review_id"].agg(set).to_dict()  # รวม id ของแต่ละสาขาเป็น set
    raw_path.parent.mkdir(parents=True, exist_ok=True)  # สร้าง data/raw ถ้ายังไม่มี

    def save(info, reviews):
        """บันทึกทันทีหลังจบแต่ละสาขา กันข้อมูลหายถ้าโปรแกรมล้มกลางทาง"""
        pd.DataFrame(reviews, columns=RAW_COLUMNS).to_csv(
            raw_path, mode="a", header=not raw_path.exists(), index=False, encoding="utf-8")  # ต่อท้าย, เขียนหัวตารางครั้งแรกเท่านั้น

    new = scrape_many(cfg, urls, known, save)  # ดึงทุกสาขา
    summary["สาขาที่สั่งดึง"] = len(urls)
    summary["รีวิวที่ดึงได้รอบนี้"] = len(new)
    if new.attrs.get("limited_view"):  # session หลุดกลางทาง
        summary["หมายเหตุ"] = "หยุดกลางทางเพราะ \"มุมมองแบบจำกัด\" ให้รัน python login.py แล้วรันใหม่"
    outputs.append(raw_path)
    if raw_path.exists():  # มีไฟล์ raw (อาจไม่มีถ้าดึงไม่ได้เลย)
        n = split_by_branch(pd.read_csv(raw_path), raw_path.parent)  # แยกไฟล์ raw ต่อสาขา ทุกคอลัมน์
        outputs.append(f"{raw_path.parent}/<ชื่อสาขา>.csv ({n} ไฟล์)")


def run_clean(cfg, in_path, summary, outputs):
    """ขั้น clean: clean + คัดรีวิวต่อสาขา เขียนไฟล์รวมและไฟล์ต่อสาขาสำหรับ Label Studio"""
    from teenoi_data.cleaner import clean_dataframe, select_per_branch  # import เมื่อใช้ (ต้องมี pythainlp)
    if not in_path.exists():  # ไม่มีรีวิวดิบให้ clean
        raise SystemExit(f"ไม่พบไฟล์รีวิวดิบ: {in_path} (ยังไม่ได้ดึงรีวิว หรือดึงไม่ได้เลย)")
    raw = pd.read_csv(in_path)  # รีวิวดิบทั้งหมด
    clean = clean_dataframe(raw, cfg["cleaning"])  # clean + ใส่ review_id
    summary["รีวิวดิบทั้งหมด"] = len(raw)
    for reason, n in clean.attrs["dropped"].items():  # จำนวนที่ตัดแต่ละขั้น
        summary[f"  ตัดออก: {reason}"] = n
    summary["รีวิวหลัง clean"] = len(clean)

    sel = cfg.get("selection") or {}  # ส่วน selection: (ไม่มีก็ได้)
    if sel.get("per_branch"):  # ตั้งโควตาต่อสาขาไว้
        clean = select_per_branch(clean, sel["per_branch"], sel.get("priority_max_rating", 4))  # คัดตามโควตา
        summary[f"รีวิวหลังคัด (สาขาละไม่เกิน {sel['per_branch']})"] = len(clean)
        for name, g in clean.groupby("branch"):  # สรุปจำนวนต่อสาขาแยกตามดาว
            stars = ", ".join(f"{int(k)}: {v}" for k, v in g["rating"].value_counts().sort_index().items())  # "1: 10, 2: 5, ..."
            summary[f"  {name}"] = f"{len(g)} (ดาว {stars})"

    out = cfg["cleaning"]["output"]  # path ของไฟล์ผลลัพธ์
    for p in (out["file"], out["full_file"]):  # สร้างโฟลเดอร์ของทั้งสองไฟล์
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    clean[["review_id", "review"]].to_csv(out["file"], index=False, encoding="utf-8")  # ไฟล์สำหรับ Label Studio
    clean.to_csv(out["full_file"], index=False, encoding="utf-8")  # ไฟล์เต็มทุกคอลัมน์
    outputs += [out["file"], out["full_file"]]
    # ไฟล์ต่อสาขาใช้ review_id เดียวกับไฟล์รวม จึงไม่ซ้ำกันข้ามสาขา import เข้า Label Studio ทีละสาขาได้
    n = split_by_branch(clean, Path(out["file"]).parent, ["review_id", "review"])
    outputs.append(f"{Path(out['file']).parent}/<ชื่อสาขา>.csv ({n} ไฟล์)")


def print_summary(summary, outputs):
    """พิมพ์สรุปตัวเลขและไฟล์ output ท้ายการรัน"""
    print("\n===== สรุป =====")
    for k, v in summary.items():  # ตามลำดับที่ใส่
        print(f"{k}: {v}")
    print("ไฟล์ output:")
    for p in outputs:
        if p:  # ข้ามค่า None (เช่น branch_finder ไม่ได้ตั้ง output_file)
            print(f"  {p if '<' in str(p) else Path(p).resolve()}")  # ข้อความ <ชื่อสาขา> พิมพ์ตรง ๆ, path จริงพิมพ์แบบเต็ม


def main():
    cfg = load_config(CONFIG, REQUIRED)  # อ่าน + ตรวจ config
    project = cfg.get("project", {})  # ส่วน project:
    log_file = setup_logging(project.get("log_level", "INFO"))  # log ลงจอ + ไฟล์
    random.seed(project.get("seed", 42))  # เวลารอแบบสุ่มซ้ำได้เหมือนเดิม
    steps, summary, outputs = cfg["steps"], {}, [log_file]  # ขั้นที่เปิด / ตัวเลขสรุป / ไฟล์ที่สร้าง

    branches = run_find_branches(cfg, summary, outputs) if steps.get("find_branches") else None  # ขั้น 1 (ถ้าเปิด)

    raw_path = Path(cfg["scraper"]["output_file"])  # ไฟล์รีวิวดิบ
    if steps.get("scrape"):  # ขั้น 2
        urls = branches["gmaps_url"].tolist() if branches is not None else read_urls(cfg["scraper"])  # ลิงก์จากขั้น 1 หรือจาก config
        run_scrape(cfg, urls, raw_path, summary, outputs)

    if steps.get("clean"):  # ขั้น 3
        in_path = raw_path if steps.get("scrape") else Path(cfg["cleaning"]["input_file"])  # เพิ่งดึง = ใช้ไฟล์ที่ดึง, ไม่ได้ดึง = ใช้ input_file
        run_clean(cfg, in_path, summary, outputs)

    print_summary(summary, outputs)


if __name__ == "__main__":  # รันตรงด้วย python main.py (ไม่รันเมื่อถูก import เช่นใน tests)
    main()
