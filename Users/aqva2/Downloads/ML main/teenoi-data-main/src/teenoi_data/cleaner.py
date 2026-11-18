"""clean ข้อความรีวิว

ห้ามตัดคำแล้วลบ stopwords ที่นี่: stopword ภาษาไทยมีคำว่า "ไม่" ทำให้ "ไม่อร่อย" กลายเป็น "อร่อย"
"""
import logging  # log จำนวนแถวที่ถูกตัดในแต่ละขั้น
import re  # regex สำหรับลบ URL/HTML/ตัวอักษรซ้ำ

from pythainlp.util import normalize  # จัดลำดับสระ/วรรณยุกต์ไทยที่พิมพ์สลับกันให้เป็นแบบมาตรฐาน

log = logging.getLogger(__name__)  # logger ชื่อ teenoi_data.cleaner

DEFAULT_CLEANING = {  # ค่าเริ่มต้น ใช้เมื่อ config ไม่ได้ระบุ key นั้น (ความหมายแต่ละ key ดูใน Data_config.yml)
    "remove_google_translation": True,  # ตัดส่วนที่ Google แปลอัตโนมัติ เก็บแค่ต้นฉบับ
    "remove_urls": True,  # ลบลิงก์
    "remove_html": True,  # ลบ tag HTML
    "normalize_thai": True,  # ใช้ pythainlp normalize
    "reduce_repeated_chars": 2,  # ตัวอักษรซ้ำเกินกี่ตัวให้ลดเหลือเท่านี้ (0/null = ไม่ลด)
    "keep_emoji": True,  # เก็บ emoji ไว้ (มีความหมายเชิงความรู้สึก)
    "min_length": 3,  # ข้อความสั้นกว่านี้ตัดทิ้ง
    "drop_duplicates": True,  # ตัดข้อความที่ซ้ำกันเป๊ะ
    "languages": ["th", "en"],  # ภาษาที่เก็บไว้ (ว่าง = ไม่กรองภาษา)
}

# emoji: บล็อก U+1F000-1FAFF (หน้ายิ้ม/สัญลักษณ์), U+2600-27BF (☀-➿), U+2B00-2BFF (ลูกศร/ดาว),
# U+FE0F (variation selector ที่ต่อท้าย emoji) และ U+200D (ZWJ ที่ใช้เชื่อม emoji หลายตัว)
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")
_THAI = re.compile("[฀-๿]")  # ช่วงตัวอักษรไทยใน Unicode
_TRANSLATED_MARK = "(แปลโดย Google)"  # ป้ายที่ Google ใส่หน้าข้อความที่แปลอัตโนมัติ
_ORIGINAL_MARK = "(ต้นฉบับ)"  # ป้ายที่ Google ใส่หน้าข้อความต้นฉบับ


def _cfg(cfg):
    """รวมค่าที่ผู้ใช้ตั้งทับค่าเริ่มต้น"""
    return {**DEFAULT_CLEANING, **(cfg or {})}  # key ใน cfg ชนะ DEFAULT_CLEANING


def clean_text(text, cfg=None):
    """clean ข้อความรีวิว 1 ข้อความตามขั้นตอนใน cfg (ลำดับขั้นมีผล ห้ามสลับ)"""
    c = _cfg(cfg)  # config ที่รวมค่าเริ่มต้นแล้ว
    text = "" if text is None or text != text else str(text)  # None/NaN -> "" (NaN เป็นค่าเดียวที่ไม่เท่ากับตัวเอง)
    if c["remove_google_translation"]:  # ขั้นที่ 1: ตัดคำแปลของ Google
        if _ORIGINAL_MARK in text:  # รูปแบบ "(แปลโดย Google) <คำแปล> (ต้นฉบับ) <ต้นฉบับ>"
            text = text.split(_ORIGINAL_MARK, 1)[1]  # เก็บเฉพาะส่วนหลัง "(ต้นฉบับ)"
        text = text.replace(_TRANSLATED_MARK, "")  # ลบป้ายที่ยังหลงเหลือ
    if c["remove_urls"]:  # ขั้นที่ 2: ลบลิงก์ http(s)://... และ www....
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # แทนด้วยช่องว่าง กันคำสองข้างติดกัน
    if c["remove_html"]:  # ขั้นที่ 3: ลบ tag เช่น <b> </b>
        text = re.sub(r"<[^>]+>", " ", text)  # แทนด้วยช่องว่างเช่นกัน
    if c["normalize_thai"]:  # ขั้นที่ 4: จัดรูปสระ/วรรณยุกต์ไทย
        text = normalize(text)  # เช่น ลบวรรณยุกต์ซ้ำ จัดลำดับสระที่พิมพ์สลับ
    n = c["reduce_repeated_chars"]  # จำนวนตัวซ้ำสูงสุดที่ยอมให้เหลือ
    if n:  # 0 หรือ None = ข้ามขั้นนี้
        # ไม่แตะตัวเลขและช่องว่าง ไม่อย่างนั้น "1000 บาท" จะกลายเป็น "100 บาท"
        text = re.sub(r"([^\d\s])\1{%d,}" % n, lambda m: m.group(1) * n, text)  # "มากกกกก" -> "มากก" (n=2)
    if not c["keep_emoji"]:  # ขั้นที่ 5 (ถ้าเลือก): ลบ emoji
        text = _EMOJI.sub(" ", text)  # แทนด้วยช่องว่าง
    text = re.sub(r"\s+", " ", text).strip()  # ช่องว่าง/ขึ้นบรรทัดหลายตัว -> ช่องว่างเดียว แล้วตัดหัวท้าย
    return re.sub(r"\s*(…|\.{3,})$", "", text)  # ตัด "…" หรือ "..." ท้ายข้อความ (Google ตัดข้อความยาว)


def detect_language(text):
    """เดาภาษาจากตัวอักษร: "th" / "en" / "other" """
    # ponytail: ดูจากชุดตัวอักษรเท่านั้น ภาษาอื่นที่ใช้อักษรละตินล้วนจะถูกนับเป็น en
    # ถ้าต้องแยกละเอียดกว่านี้ค่อยเปลี่ยนไปใช้ไลบรารีตรวจภาษา
    if _THAI.search(text):  # มีอักษรไทยแม้ตัวเดียว (รีวิวผสมไทย-อังกฤษ) นับเป็นไทย
        return "th"
    if any(ch.isalpha() and not ch.isascii() for ch in text):  # มีตัวอักษรที่ไม่ใช่ ASCII เช่น ญี่ปุ่น จีน
        return "other"
    return "en" if re.search(r"[A-Za-z]", text) else "other"  # มีอักษรอังกฤษ = en, ไม่มีเลย (emoji ล้วน) = other


def select_per_branch(df, per_branch, priority_max_rating=4):
    """คัดสาขาละไม่เกิน per_branch แถว: ดาว <= priority_max_rating ก่อน ที่เหลือเติมด้วยดาวสูงกว่า

    กลุ่มเติมสลับดาวทีละแถว (เช่น 5, 4, 5, 4, ...) จึงได้แต่ละดาวเท่า ๆ กัน ถ้าดาวไหนหมดก่อน ที่เหลือเป็นดาวอื่น
    ทุกกลุ่มเลือกใหม่สุดก่อน: ในไฟล์ raw รอบดึงหลัง (scraped_at มากกว่า) ใหม่กว่ารอบก่อน
    และแถวในรอบเดียวกันเรียงใหม่ -> เก่าอยู่แล้ว  review_id คงเดิม (มีช่องว่างได้) เพื่อไม่ให้เลขเปลี่ยนเมื่อรันซ้ำ
    """
    order = df.sort_values("scraped_at", ascending=False, kind="stable")  # รอบดึงล่าสุดก่อน (stable = คงลำดับเดิมในรอบเดียวกัน)
    fill = ~(order["rating"] <= priority_max_rating)  # True = กลุ่มเติม (ดาวสูง หรือไม่มีดาว)
    # turn: กลุ่มดาวต่ำ = -1 (มาก่อนทั้งหมด), กลุ่มเติม = ลำดับที่ของแถวนั้นภายในดาวเดียวกันของสาขา (0, 1, 2, ...)
    turn = order.groupby(["branch_id", "rating"], dropna=False).cumcount().where(fill, -1)
    order = order.assign(_turn=turn).sort_values("_turn", kind="stable")  # เรียงตาม turn: ดาวต่ำก่อน แล้วสลับดาวทีละแถว
    keep = order.groupby("branch_id", sort=False).head(per_branch).index  # หัวตาราง per_branch แถวของแต่ละสาขา
    return df.loc[keep.sort_values()]  # คืนตามลำดับเดิมของ df (เรียงตาม index)


def clean_dataframe(df, cfg=None):
    """clean คอลัมน์ review แล้วกรองแถว ใส่ review_id 1, 2, 3, ...

    จำนวนแถวที่ถูกตัดแต่ละขั้นอยู่ใน df.attrs["dropped"]
    """
    c = _cfg(cfg)  # config ที่รวมค่าเริ่มต้นแล้ว
    df = df.copy()  # ไม่แก้ DataFrame ของผู้เรียก
    df["review"] = df["review"].map(lambda t: clean_text(t, c))  # clean ทีละข้อความ
    dropped = {}  # เหตุผล -> จำนวนแถวที่ถูกตัด

    def keep(mask, reason):
        """เก็บเฉพาะแถวที่ mask เป็น True และจดจำนวนที่ถูกตัด"""
        nonlocal df  # แก้ตัวแปร df ของฟังก์ชันด้านนอก
        dropped[reason] = int((~mask).sum())  # นับแถวที่ไม่ผ่าน
        log.info("ตัด %d แถว: %s (เหลือ %d)", dropped[reason], reason, int(mask.sum()))  # บอกใน log
        df = df[mask]  # กรองทิ้ง

    min_len = c["min_length"]  # ความยาวขั้นต่ำ
    keep(df["review"].str.len() >= min_len, f"สั้นกว่า {min_len} ตัวอักษร (รวมรีวิวที่ไม่มีข้อความ)")  # รีวิวที่ให้แต่ดาวจะเป็น "" และถูกตัดตรงนี้
    if c["languages"]:  # กรองภาษาเมื่อระบุรายการภาษา
        keep(df["review"].map(detect_language).isin(c["languages"]), f"ภาษาไม่อยู่ใน {c['languages']}")
    if c["drop_duplicates"]:  # ตัดข้อความซ้ำ เก็บแถวแรกที่เจอ
        keep(~df["review"].duplicated(), "ข้อความซ้ำ")

    df = df.reset_index(drop=True)  # index ใหม่ 0, 1, 2, ... หลังกรอง
    df.insert(0, "review_id", range(1, len(df) + 1))  # review_id เริ่มที่ 1 เป็นคอลัมน์แรก
    df.attrs["dropped"] = dropped  # แนบสถิติไว้ให้ main.py พิมพ์สรุป
    return df
