"""ดึงรีวิวจาก Google Maps (รองรับการดึงแบบเพิ่มเติมด้วย stop_at_ids)"""
import logging  # log ความคืบหน้าและคำเตือน
import random  # สุ่มเวลารอ ไม่ให้จังหวะเหมือนบอทเกินไป
import re  # regex แกะ URL/ตัวเลข/วันที่
import time  # time.sleep รอหน้าเว็บตอบสนอง
from datetime import datetime, timedelta  # คำนวณวันที่จาก "3 เดือนที่แล้ว"
from urllib.parse import unquote  # แปลง %E0%B8... ใน URL กลับเป็นภาษาไทย

import pandas as pd  # รวมรีวิวเป็นตาราง

from .browser import open_context  # เปิด Chrome พร้อม session ที่ล็อกอินไว้

log = logging.getLogger(__name__)  # logger ชื่อ teenoi_data.scraper

# CSS selector ของหน้า Google Maps: Google เปลี่ยน class name บ่อย ถ้าดึงได้ 0 รีวิวให้แก้ที่นี่ที่เดียว
SEL = {
    "card": "div.jftiEf[data-review-id]",  # การ์ดรีวิว 1 ใบ (มี id ของรีวิว)
    "more": "button.w8nwRe[aria-expanded='false']",  # ปุ่ม "เพิ่มเติม" ที่ยังไม่ได้กด (ข้อความถูกตัด)
    "text": "span.wiI7pd",  # ข้อความรีวิว
    "stars": "span.kvMYJc[role='img']",  # ดาว (จำนวนอยู่ใน aria-label เช่น "4 ดาว")
    "date": "span.rsqaWe",  # วันที่แบบ "3 เดือนที่แล้ว"
    "name": "h1",  # ชื่อร้าน
    "tab": "button[role='tab']",  # แท็บด้านบน (ภาพรวม / รีวิว / ...)
    "place_link": "a[href*='/maps/place/']",  # ลิงก์ร้านในหน้าผลค้นหา (branch_finder ใช้)
    # สามตัวล่างยังไม่ได้ทดสอบ ถ้าหาไม่เจอจะได้ค่า None (ไม่ทำให้การดึงรีวิวพัง)
    "address": "button[data-item-id='address']",  # ปุ่มที่อยู่ (ที่อยู่อยู่ใน aria-label)
    "place_rating": "div.F7nice span[aria-hidden='true']",  # คะแนนเฉลี่ยของร้าน เช่น 4.3
    "place_review_count": "div.F7nice",  # กล่องที่มีจำนวนรีวิว เช่น "(1,234)"
}
# ข้อความบนหน้าเว็บที่ใช้หาปุ่ม (ต้องตรงกับ browser.locale: th-TH)
TXT = {
    "tab_reviews": "รีวิว",  # ชื่อแท็บรีวิว
    "sort_button": "เกี่ยวข้องที่สุด",  # ปุ่มเปิดเมนูเรียงลำดับ (แสดงค่าปัจจุบัน)
    "limited_view": "มุมมองแบบจำกัด",  # ข้อความที่ขึ้นเมื่อยังไม่ได้ล็อกอิน
}
# sort_by -> ข้อความในเมนูเรียงลำดับ (most_relevant คือค่าเริ่มต้น ไม่ต้องกด)
SORT_MENU = {"newest": "ใหม่ที่สุด", "lowest_rating": "คะแนนต่ำสุด", "highest_rating": "คะแนนสูงสุด"}
RAW_COLUMNS = [  # คอลัมน์ของ reviews_raw.csv ตามลำดับ
    "branch_id", "branch", "rating", "date_text", "date_approx",  # สาขา ดาว วันที่
    "review", "gmaps_review_id", "source_url", "scraped_at",  # ข้อความ id ลิงก์ เวลาที่ดึง
]
KEEP_CARDS = 30  # การ์ดท้ายที่เก็บไว้ในหน้า: อ่านซ้ำได้อีก 2-3 รอบ เผื่อข้อความที่กด "เพิ่มเติม" แล้วขยายไม่ทัน
PROGRESS_EVERY = 500  # log ความคืบหน้าทุก ๆ กี่การ์ด
UNKNOWN_RATING = 9  # ใช้แทนดาวที่อ่านไม่ออก ให้มากกว่าทุก priority_max_rating จึงไม่ถูกนับเป็นดาวต่ำ

_DAYS = {"นาที": 0, "ชั่วโมง": 0, "วัน": 1, "สัปดาห์": 7, "เดือน": 30, "ปี": 365}  # หน่วยเวลา -> จำนวนวันโดยประมาณ

# JavaScript ที่รันในหน้าเว็บผ่าน page.evaluate
# กดปุ่ม "เพิ่มเติม" ทุกปุ่มให้ข้อความเต็ม คืนจำนวนปุ่มที่กด
_JS_EXPAND = "sel => { const b = document.querySelectorAll(sel); b.forEach(x => x.click()); return b.length; }"
_JS_CARDS = """sel => [...document.querySelectorAll(sel.card)].map(c => ({
    // อ่านการ์ดทุกใบในหน้าเป็น {id, text, stars, date}
    id: c.getAttribute('data-review-id'),
    // ?. กันการ์ดที่ไม่มีส่วนนั้น (เช่น รีวิวที่ให้แต่ดาว) ไม่ให้ error
    text: c.querySelector(sel.text)?.innerText || '',
    stars: c.querySelector(sel.stars)?.getAttribute('aria-label') || '',
    date: c.querySelector(sel.date)?.innerText || '',
}))"""
# ลบการ์ดเก่า (และเส้นคั่นที่ตามหลัง) ออกจากหน้า เหลือ keep ใบท้าย คืนจำนวนการ์ดที่เหลือ
# ถ้าปล่อยการ์ดสะสมในหน้า ยิ่งเลื่อนลึกยิ่งช้า (วัดได้: 500 ใบที่สี่ช้ากว่า 500 ใบแรก ~8 เท่า)
_JS_PRUNE = """([sel, keep]) => {
    const all = [...document.querySelectorAll(sel)];
    // การ์ดทั้งหมดยกเว้น keep ใบท้าย
    for (const c of all.slice(0, Math.max(0, all.length - keep))) {
        // ตัวถัดไปที่ไม่ใช่การ์ด คือเส้นคั่น ลบไปด้วย
        const s = c.nextElementSibling;
        if (s && !s.matches(sel)) s.remove();
        c.remove();
    }
    return document.querySelectorAll(sel).length;
}"""
# true เมื่อมีการ์ดใหม่โหลดเข้ามาแล้ว (จำนวนการ์ดมากกว่า n)
_JS_MORE_CARDS = "([sel, n]) => document.querySelectorAll(sel).length > n"
# เลื่อน panel รีวิวลงล่างสุด: ไล่จากการ์ดขึ้นไปหา element แรกที่เลื่อนได้ (ไม่ต้องพึ่ง class ของ panel)
_JS_SCROLL = """sel => {
    let el = document.querySelector(sel);
    while (el) {
        const o = getComputedStyle(el).overflowY;
        // element นี้มี scrollbar: เลื่อนลงสุดแล้วจบ
        if (o === 'auto' || o === 'scroll') { el.scrollTo(0, el.scrollHeight); return true; }
        el = el.parentElement;
    }
    return false;
}"""


class LimitedViewError(RuntimeError):
    """Google Maps เปิดแบบ "มุมมองแบบจำกัด" (ยังไม่ได้ล็อกอิน) เห็นรีวิวแค่ ~5 รีวิว"""


def approx_date(date_text, now=None):
    """ "3 เดือนที่แล้ว" -> วันที่โดยประมาณ (YYYY-MM-DD) อ่านไม่ออกคืน None"""
    m = re.search(r"(\d+)?\s*(นาที|ชั่วโมง|วัน|สัปดาห์|เดือน|ปี)", date_text or "")  # จับ "ตัวเลข(ถ้ามี) + หน่วย"
    if not m:  # ไม่มีหน่วยเวลาที่รู้จัก
        return None
    days = int(m.group(1) or 1) * _DAYS[m.group(2)]  # ไม่มีตัวเลข ("สัปดาห์ที่แล้ว") = 1 หน่วย
    return ((now or datetime.now()) - timedelta(days=days)).date().isoformat()  # ย้อนจากเวลาปัจจุบัน


def parse_place_url(url):
    """ดึง name, branch_id, lat, lng จาก URL แบบ /maps/place/<name>/...!1s0x..:0x..!3d<lat>!4d<lng>"""
    name = re.search(r"/maps/place/([^/@?]+)", url)  # ชื่อร้าน (encode แบบ URL)
    bid = re.search(r"!1s(0x[0-9a-fA-F]+:0x[0-9a-fA-F]+)", url)  # id ถาวรของสถานที่ (ใช้เป็น branch_id)
    lat = re.search(r"!3d(-?\d+(?:\.\d+)?)", url)  # ละติจูด
    lng = re.search(r"!4d(-?\d+(?:\.\d+)?)", url)  # ลองจิจูด
    return {
        "name": unquote(name.group(1)).replace("+", " ") if name else None,  # %E0%B8.. -> ไทย, + -> ช่องว่าง
        "branch_id": bid.group(1) if bid else None,  # ลิงก์สั้นยังไม่มี id จะได้ None
        "lat": float(lat.group(1)) if lat else None,
        "lng": float(lng.group(1)) if lng else None,
    }


def _try(fn):
    """เรียก fn() ถ้า error คืน None (ใช้กับข้อมูลเสริมที่หาไม่เจอก็ไม่เป็นไร)"""
    try:
        return fn()
    except Exception:  # selector หาไม่เจอ / timeout
        return None


def _num(text):
    """ตัวเลขตัวแรกในข้อความ เช่น "4.3" -> 4.3, "1,234" -> 1234.0 ไม่เจอคืน None"""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text or "")  # ตัวเลขที่อาจมี , คั่นหลักพันและทศนิยม
    return float(m.group().replace(",", "")) if m else None  # ลบ , ก่อนแปลง


def _rating(card):
    """จำนวนดาวจาก aria-label เช่น "4 ดาว" -> 4 อ่านไม่ออกคืน None"""
    m = re.search(r"\d", card["stars"])  # ดาวมีหลักเดียว (1-5)
    return int(m.group()) if m else None


def _read_place_info(page, url):
    """อ่านข้อมูลร้านจากหน้าที่เปิดอยู่: name, branch_id, lat, lng, gmaps_url, address, google_rating, review_count"""
    info = parse_place_url(page.url)  # URL เต็มหลัง redirect มี id/พิกัด
    info["branch_id"] = info["branch_id"] or url  # ไม่มี id ใน URL ใช้ลิงก์ที่ได้มาเป็น id แทน
    info["name"] = _try(lambda: page.locator(SEL["name"]).first.inner_text(timeout=5000).strip()) or info["name"]  # ชื่อจาก h1 ดีกว่าจาก URL
    info["gmaps_url"] = page.url  # เก็บลิงก์เต็ม
    addr = _try(lambda: page.locator(SEL["address"]).first.get_attribute("aria-label", timeout=3000))  # "ที่อยู่: ..."
    info["address"] = addr.split(":", 1)[-1].strip() if addr else None  # ตัดคำว่า "ที่อยู่:" ออก
    info["google_rating"] = _num(_try(lambda: page.locator(SEL["place_rating"]).first.inner_text(timeout=3000)))  # คะแนนเฉลี่ย
    cnt = _try(lambda: page.locator(SEL["place_review_count"]).first.inner_text(timeout=3000))  # ข้อความที่มี "(1,234)"
    cnt = re.search(r"\(([\d,]+)\)", cnt or "")  # ตัวเลขในวงเล็บ
    info["review_count"] = int(cnt.group(1).replace(",", "")) if cnt else None  # จำนวนรีวิวทั้งหมดของร้าน
    return info


def _apply_sort(page, sort_by):
    """กดเรียงรีวิวตาม sort_by คืน True ถ้าไม่ต้องเรียง (most_relevant) หรือเรียงสำเร็จ"""
    if sort_by not in SORT_MENU:  # most_relevant เป็นลำดับเริ่มต้นอยู่แล้ว
        return True
    try:
        page.locator("button", has_text=TXT["sort_button"]).first.click(timeout=10000)  # เปิดเมนูเรียง
        page.locator("[role='menuitemradio']", has_text=SORT_MENU[sort_by]).first.click(timeout=10000)  # เลือกตัวเลือก
        time.sleep(2)  # รอรายการรีวิวโหลดใหม่ตามลำดับที่เลือก
        return True
    except Exception:  # หาปุ่ม/ตัวเลือกไม่เจอ
        return False


def _to_rows(found, info, url, max_reviews):
    """แปลงการ์ดที่เก็บได้เป็นแถวตาม RAW_COLUMNS (ตัดเมื่อรีวิวที่มีข้อความครบ max_reviews)"""
    now = datetime.now()  # ใช้เวลาเดียวกันทั้งรอบ ทั้ง date_approx และ scraped_at
    rows, n_text = [], 0  # n_text = จำนวนรีวิวที่มีข้อความที่ใส่ไปแล้ว
    for c in found.values():  # ตามลำดับที่เจอบนหน้า (ใหม่ -> เก่า เมื่อเรียงแบบ newest)
        text = c["text"].strip()  # ข้อความรีวิว ("" = ให้แต่ดาว)
        if text:  # นับเฉพาะรีวิวที่มีข้อความ
            if n_text >= max_reviews:  # ครบเพดานแล้ว หยุด
                break
            n_text += 1
        rows.append({  # รีวิวที่ไม่มีข้อความก็เก็บ (ไม่นับเพดาน) cleaner จะตัดทิ้งเอง
            "branch_id": info["branch_id"],
            "branch": info["name"],
            "rating": _rating(c),
            "date_text": c["date"],  # ข้อความวันที่ดิบ
            "date_approx": approx_date(c["date"], now),  # วันที่โดยประมาณ
            "review": text,
            "gmaps_review_id": c["id"],  # id ของ Google ใช้กันดึงซ้ำ
            "source_url": url,  # ลิงก์ที่สั่งดึง (อาจเป็นลิงก์สั้น)
            "scraped_at": now.isoformat(timespec="seconds"),  # เวลาที่ดึง ใช้เรียงใหม่/เก่าตอนคัด
        })
    return rows


def scrape_place(page, url, max_reviews, stop_at_ids=None, *, sort_by="newest", max_idle_rounds=8, delay=(1.2, 2.5),
                 priority_target=None, priority_max_rating=4):
    """ดึงรีวิวของ 1 สาขา คืน (place_info, reviews)

    stop_at_ids: ชุด gmaps_review_id ที่เคยดึงแล้ว ถ้าเรียงตามใหม่สุดแล้วเจอ id ในชุดนี้จะหยุดทันที
    max_reviews: นับเฉพาะรีวิวที่มีข้อความ (รีวิวที่ให้แต่ดาวยังถูกเก็บ แต่ไม่นับ)
    priority_target: หยุดเมื่อได้รีวิวที่มีข้อความและดาว <= priority_max_rating ครบจำนวนนี้ (None = ไม่ใช้)
    """
    stop_at_ids = stop_at_ids or set()  # None -> ชุดว่าง
    page.goto(url, wait_until="domcontentloaded", timeout=60000)  # เปิดหน้าร้าน (รอแค่ HTML ไม่รอรูป/สคริปต์ทั้งหมด)
    page.wait_for_selector(SEL["name"], timeout=30000)  # รอชื่อร้านขึ้น = หน้าร้านพร้อม
    _try(lambda: page.wait_for_url(re.compile(r"!1s0x"), timeout=15000))  # รอ URL เต็มหลัง redirect จากลิงก์สั้น

    info = _read_place_info(page, url)  # ชื่อ id ที่อยู่ คะแนน
    if TXT["limited_view"] in page.content():  # ยังไม่ได้ล็อกอิน เห็นรีวิวแค่ไม่กี่อัน
        raise LimitedViewError(f"Google Maps เปิดแบบ \"{TXT['limited_view']}\" ให้รัน python login.py เพื่อล็อกอินก่อน")

    page.locator(SEL["tab"], has_text=TXT["tab_reviews"]).first.click(timeout=15000)  # เปิดแท็บรีวิว
    time.sleep(1.5)  # รอแท็บรีวิวโหลด
    sorted_ok = _apply_sort(page, sort_by)  # เรียงตามที่ตั้งไว้
    if not sorted_ok and sort_by != "newest":  # รอบเสริม (เช่น คะแนนต่ำสุด) ถ้าเรียงไม่ได้ให้ข้าม ไม่งั้นจะได้รีวิวผิดกลุ่ม
        log.warning("เรียงตาม \"%s\" ไม่สำเร็จ ข้ามรอบนี้: %s", SORT_MENU[sort_by], info["name"])
        return info, []
    if not sorted_ok:  # รอบ newest เรียงไม่ได้: ยังดึงต่อด้วยลำดับเริ่มต้น
        log.warning("เรียงตาม \"%s\" ไม่สำเร็จ ใช้ลำดับเริ่มต้นแทน: %s", SORT_MENU[sort_by], info["name"])
    newest = sort_by == "newest" and sorted_ok  # เรียงใหม่สุดจริงเท่านั้นจึงหยุดเมื่อเจอรีวิวเดิมได้

    if not _try(lambda: page.wait_for_selector(SEL["card"], timeout=15000)):  # ไม่มีการ์ดรีวิวขึ้นเลย
        log.warning("ไม่พบการ์ดรีวิวของ %s (ถ้าเป็นทุกสาขา ให้ตรวจ SEL ใน scraper.py)", info["name"])
        return info, []

    found, seen, idle = {}, set(), 0  # found: id -> card (dict คงลำดับที่เจอ), seen: ทุก id ที่เลื่อนผ่าน, idle: รอบที่ไม่มีอะไรใหม่
    while True:  # เลื่อนไปเรื่อย ๆ จนเข้าเงื่อนไขหยุดข้อใดข้อหนึ่ง
        if page.evaluate(_JS_EXPAND, SEL["more"]):  # มีปุ่ม "เพิ่มเติม" ให้กด
            time.sleep(0.6)  # รอข้อความขยาย
        n_seen, hit_known = len(seen), False  # จำนวนก่อนอ่านรอบนี้ / เจอรีวิวเดิมหรือยัง
        for c in page.evaluate(_JS_CARDS, SEL):  # ในหน้าเหลือแค่การ์ดท้าย ๆ เพราะลบของเก่าออกทุกรอบ
            seen.add(c["id"])  # จดว่าเลื่อนผ่านแล้ว
            if c["id"] in stop_at_ids:  # รีวิวที่เคยดึงไปแล้ว
                hit_known = True
            elif c["id"] not in found or len(c["text"]) > len(found[c["id"]]["text"]):  # ใหม่ หรือข้อความยาวขึ้น (เพิ่งขยาย)
                found[c["id"]] = c  # เก็บข้อความที่ยาวที่สุดของแต่ละ review id
        if hit_known and newest:  # เรียงใหม่สุดแล้วเจอของเดิม = ที่เหลือเก่ากว่าและมีแล้วทั้งหมด
            break
        with_text = [c for c in found.values() if c["text"].strip()]  # เฉพาะรีวิวที่มีข้อความ
        if len(with_text) >= max_reviews:  # ครบเพดานต่อสาขา
            break
        n_pri = sum(1 for c in with_text if (_rating(c) or UNKNOWN_RATING) <= priority_max_rating)  # จำนวนรีวิวดาวต่ำ
        if priority_target and n_pri >= priority_target:  # ได้รีวิวดาวต่ำครบเป้า
            break
        if len(seen) // PROGRESS_EVERY > n_seen // PROGRESS_EVERY:  # สาขาใหญ่ใช้เวลานาน บอกความคืบหน้าเมื่อข้ามหลัก 500
            log.info("  %s: เลื่อนถึง %d รีวิว (มีข้อความ %d, ดาว <= %d: %d)",
                     info["name"], len(seen), len(with_text), priority_max_rating, n_pri)
        idle = idle + 1 if len(seen) == n_seen else 0  # รีวิวที่เคยดึงแล้วก็ถือว่าคืบหน้า
        if idle >= max_idle_rounds:  # เลื่อนแล้วไม่มีอะไรใหม่หลายรอบติด = สุดรายการ
            break
        n_dom = page.evaluate(_JS_PRUNE, [SEL["card"], KEEP_CARDS])  # ลบการ์ดเก่าออกจากหน้า กันช้า
        page.evaluate(_JS_SCROLL, SEL["card"])  # เลื่อนลงให้ Google โหลดชุดถัดไป
        time.sleep(0.3)  # ให้หน้าเริ่มโหลด
        # ไปต่อทันทีที่การ์ดชุดใหม่โหลดเสร็จ delay เป็นเวลารอสูงสุด ไม่ใช่เวลารอตายตัว
        _try(lambda: page.wait_for_function(_JS_MORE_CARDS, arg=[SEL["card"], n_dom], timeout=random.uniform(*delay) * 1000))

    return info, _to_rows(found, info, url, max_reviews)  # แปลงเป็นแถวพร้อมบันทึก


def scrape_many(cfg, urls, known_ids_by_branch=None, on_place_done=None):
    """ดึงรีวิวหลายสาขา คืน DataFrame คอลัมน์ RAW_COLUMNS

    known_ids_by_branch: {branch_id: set(gmaps_review_id)} รีวิวที่เคยดึงแล้ว (ดึงเฉพาะที่ใหม่กว่า)
    on_place_done(place_info, reviews): เรียกหลังจบแต่ละสาขา ใช้บันทึกผลทันที
    ถ้าเจอ "มุมมองแบบจำกัด" จะหยุดและตั้ง df.attrs["limited_view"] = True
    """
    s = cfg.get("scraper", {})  # ส่วน scraper: ใน config
    d = s.get("delay", {})  # ส่วน scraper.delay:
    priority_max_rating = (cfg.get("selection") or {}).get("priority_max_rating", 4)  # ดาว <= ค่านี้ = กลุ่มดาวต่ำ
    # review id ไม่ซ้ำกันข้ามสาขา จึงรวมเป็นชุดเดียวได้ (ลิงก์สั้นยังไม่รู้ branch_id ก่อนเปิดหน้า)
    stop = set().union(*(known_ids_by_branch or {}).values())
    # sort_by เป็นค่าเดียวหรือ list ก็ได้ (ดึงสาขาละหลายรอบ รอบหลังข้ามรีวิวที่รอบก่อนได้แล้ว)
    sorts = s.get("sort_by", "newest")
    sorts = [sorts] if isinstance(sorts, str) else sorts  # ทำให้เป็น list เสมอ
    sorts = sorted(sorts, key=lambda x: x != "newest")  # newest ต้องมาก่อน เพราะรอบ newest หยุดทันทีเมื่อเจอรีวิวที่รู้จัก
    rows, limited = [], False  # รีวิวทั้งหมด / เจอมุมมองแบบจำกัดหรือไม่
    with open_context(cfg) as ctx:  # เปิด browser ครั้งเดียวใช้ทุกสาขา
        page = ctx.pages[0] if ctx.pages else ctx.new_page()  # persistent context มักมีแท็บเปิดอยู่แล้ว 1 แท็บ
        for i, (url, sort_by) in enumerate((u, x) for u in urls for x in sorts):  # ทุกคู่ (สาขา, วิธีเรียง)
            if i:  # ไม่ต้องรอก่อนรอบแรก
                time.sleep(random.uniform(d.get("between_places_min", 4), d.get("between_places_max", 8)))  # พักระหว่างรอบ
            try:
                info, reviews = scrape_place(
                    page, url, s.get("max_reviews_per_place", 1000), stop,  # หน้า ลิงก์ เพดาน รีวิวที่รู้จักแล้ว
                    sort_by=sort_by,  # วิธีเรียงของรอบนี้
                    max_idle_rounds=s.get("max_idle_rounds", 8),  # เลื่อนไม่เจออะไรใหม่กี่รอบแล้วหยุด
                    delay=(d.get("scroll_min", 1.2), d.get("scroll_max", 2.5)),  # ช่วงเวลารอหลังเลื่อน
                    # เป้าดาวต่ำใช้กับรอบ newest เท่านั้น (รอบอื่นเรียงตามดาวอยู่แล้ว)
                    priority_target=s.get("priority_target") if sort_by == "newest" else None,
                    priority_max_rating=priority_max_rating,
                )
            except LimitedViewError as e:  # session หลุด สาขาต่อ ๆ ไปก็จะเจอเหมือนกัน
                log.warning("%s", e)
                limited = True
                break  # หยุดทั้งหมด
            except Exception:  # error อื่นของสาขานี้ (timeout, หน้าเปลี่ยน ฯลฯ)
                log.exception("ดึงรีวิวไม่สำเร็จ ข้ามสาขานี้: %s", url)  # log พร้อม traceback
                continue  # ไปสาขาถัดไป
            log.info("[%d/%d] %s (%s): ได้ %d รีวิว", i // len(sorts) + 1, len(urls), info["name"], sort_by, len(reviews))
            stop.update(r["gmaps_review_id"] for r in reviews)  # รอบถัดไปไม่เก็บรีวิวเหล่านี้ซ้ำ
            rows.extend(reviews)  # สะสมผล
            if on_place_done:  # ผู้เรียกอยากบันทึกทันที
                on_place_done(info, reviews)
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)  # ตารางรีวิวรอบนี้ (ว่างก็ยังมีคอลัมน์ครบ)
    df.attrs["limited_view"] = limited  # แจ้ง main.py ว่าหยุดกลางทางหรือไม่
    return df
