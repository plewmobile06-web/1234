import base64
import json
from typing import List, Optional

import anthropic
from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel, ValidationError

router = APIRouter(prefix="/api/v1", tags=["อ่านเอกสารผลการเรียนด้วย AI"])

# ใช้ Claude อ่านภาพ/PDF ใบแสดงผลการเรียน แล้วส่งค่ากลับไปให้ผู้ใช้ตรวจสอบในฟอร์มก่อนบันทึก
CLAUDE_MODEL = "claude-opus-5-5"
MAX_FILES = 5
MAX_IMAGE_BYTES = 5 * 1024 * 1024    # ขนาดภาพสูงสุดที่ Claude รับต่อภาพ
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_TERMS = 8

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
PDF_TYPE = "application/pdf"

_client = None


def ไคลเอนต์Claude():
    # สร้างเมื่อใช้งานครั้งแรก ระบบส่วนอื่นจึงยังทำงานได้แม้ยังไม่ได้ตั้งค่า API key
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


class ผลการอ่านภาค(BaseModel):
    term_label: str
    gpax: Optional[float]
    fw_count: Optional[int]
    midterm_score: Optional[float]


class ผลการอ่านเอกสาร(BaseModel):
    student_id: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    credits_earned: Optional[int]
    terms: List[ผลการอ่านภาค]
    notes: str


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "student_id": {"type": ["string", "null"]},
        "first_name": {"type": ["string", "null"]},
        "last_name": {"type": ["string", "null"]},
        "credits_earned": {"type": ["integer", "null"]},
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term_label": {"type": "string"},
                    "gpax": {"type": ["number", "null"]},
                    "fw_count": {"type": ["integer", "null"]},
                    "midterm_score": {"type": ["number", "null"]},
                },
                "required": ["term_label", "gpax", "fw_count", "midterm_score"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["student_id", "first_name", "last_name", "credits_earned", "terms", "notes"],
    "additionalProperties": False,
}

EXTRACTION_PROMPT = """เอกสารที่แนบมาเป็นเอกสารผลการเรียนของนักศึกษาปริญญาตรีในประเทศไทย (เช่น ใบแสดงผลการเรียน/Transcript หรือใบรายงานคะแนน) อาจมีหลายหน้า ให้ดึงข้อมูลต่อไปนี้

- student_id, first_name, last_name: รหัสและชื่อนักศึกษา (ชื่อไม่รวมคำนำหน้า เช่น นาย นางสาว)
- credits_earned: หน่วยกิตสะสมที่สอบผ่านแล้วล่าสุด (CR. EARNED / หน่วยกิตสะสม)
- terms: ภาคเรียนปกติเรียงจากภาคแรกไปภาคล่าสุด ภาคละ 1 รายการ
  - term_label: ชื่อภาคตามเอกสาร เช่น "1/2567"
  - gpax: เกรดเฉลี่ยสะสม ณ สิ้นภาคนั้น (GPAX / CUM. GPA / เกรดเฉลี่ยสะสม) ไม่ใช่เกรดเฉลี่ยรายภาค (GPA / GPS)
  - fw_count: จำนวนรายวิชาในภาคนั้นที่ได้เกรด F หรือ W
  - midterm_score: คะแนนเก็บหรือคะแนนกลางภาครวมเป็นเปอร์เซ็นต์ (0-100) ถ้าเอกสารมีให้

กติกา
- ห้ามเดาหรือคำนวณค่าที่ไม่มีในเอกสาร ถ้าไม่มีหรืออ่านไม่ชัด ให้ใส่ null
- ภาคฤดูร้อนไม่ต้องใส่เป็นภาคแยก แต่ให้ระบุใน notes ว่ามีภาคฤดูร้อนภาคใด
- notes: สรุปสั้น ๆ เป็นภาษาไทย ว่าเป็นเอกสารประเภทใด และค่าใดอ่านไม่ได้หรือไม่แน่ใจ เพื่อให้อาจารย์ตรวจสอบ
- ถ้าเอกสารไม่ใช่ผลการเรียน ให้ terms เป็นรายการว่าง และอธิบายใน notes"""


def สร้างบล็อกเอกสาร(content_type, data):
    encoded = base64.standard_b64encode(data).decode("utf-8")
    if content_type == PDF_TYPE:
        return {"type": "document", "source": {"type": "base64", "media_type": PDF_TYPE, "data": encoded}}
    return {"type": "image", "source": {"type": "base64", "media_type": content_type, "data": encoded}}


def ตรวจค่าที่อ่านได้(term):
    """ตัดค่าที่อยู่นอกช่วงที่เป็นไปได้ทิ้ง (เป็น null) ให้ผู้ใช้กรอกเอง แทนการส่งค่าผิดเข้าโมเดล"""
    gpax = term.gpax if term.gpax is not None and 0 <= term.gpax <= 4 else None
    fw = term.fw_count if term.fw_count is not None and 0 <= term.fw_count <= 15 else None
    mid = term.midterm_score if term.midterm_score is not None and 0 <= term.midterm_score <= 100 else None
    return {"term_label": term.term_label, "gpax": gpax, "fw_count": fw, "midterm_score": mid}


@router.post("/extract-transcript", summary="อ่านภาพ/PDF เอกสารผลการเรียนด้วย AI แล้วส่งค่ากลับมาให้ตรวจสอบ")
def อ่านเอกสารผลการเรียน(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="กรุณาแนบไฟล์อย่างน้อย 1 ไฟล์")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"แนบได้สูงสุด {MAX_FILES} ไฟล์ต่อครั้ง")

    content = []
    for f in files:
        data = f.file.read()
        content_type = (f.content_type or "").lower()
        if content_type == "image/jpg":
            content_type = "image/jpeg"
        if content_type in IMAGE_TYPES:
            if len(data) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=400, detail=f"ไฟล์ {f.filename} ใหญ่เกิน 5 MB กรุณาย่อขนาดภาพ")
        elif content_type == PDF_TYPE:
            if len(data) > MAX_PDF_BYTES:
                raise HTTPException(status_code=400, detail=f"ไฟล์ {f.filename} ใหญ่เกิน 20 MB")
        else:
            raise HTTPException(status_code=400, detail=f"ไฟล์ {f.filename} ไม่รองรับ (ใช้ได้: JPG, PNG, GIF, WEBP, PDF)")
        content.append(สร้างบล็อกเอกสาร(content_type, data))
    content.append({"type": "text", "text": EXTRACTION_PROMPT})

    try:
        response = ไคลเอนต์Claude().beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
            # ถ้าโมเดลหลักปฏิเสธคำขอ ให้ API ส่งต่อโมเดลสำรองอัตโนมัติ
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=503, detail="ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY หรือ API key ไม่ถูกต้อง")
    except anthropic.BadRequestError as e:
        raise HTTPException(status_code=400, detail=f"AI ไม่สามารถอ่านไฟล์นี้ได้: {e.message}")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="มีการเรียกใช้ AI ถี่เกินไป กรุณาลองใหม่อีกครั้งในอีกสักครู่")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"บริการ AI ขัดข้อง ({e.status_code}) กรุณาลองใหม่")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=502, detail="เชื่อมต่อบริการ AI ไม่ได้ กรุณาตรวจสอบอินเทอร์เน็ต")
    except TypeError as e:
        # SDK ไม่พบข้อมูลยืนยันตัวตนเลย (ไม่มี API key / profile)
        if "auth" in str(e).lower() or "api_key" in str(e).lower():
            raise HTTPException(status_code=503, detail="ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY ในเครื่องที่รัน server")
        raise

    if response.stop_reason == "refusal":
        raise HTTPException(status_code=422, detail="AI ปฏิเสธการอ่านเอกสารนี้ กรุณากรอกข้อมูลเอง")
    if response.stop_reason == "max_tokens":
        raise HTTPException(status_code=502, detail="เอกสารยาวเกินไป ลองแนบเฉพาะหน้าที่มีผลการเรียน")

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        parsed = ผลการอ่านเอกสาร.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        raise HTTPException(status_code=502, detail="อ่านผลจาก AI ไม่สำเร็จ กรุณาลองใหม่หรือกรอกข้อมูลเอง")

    terms = [ตรวจค่าที่อ่านได้(t) for t in parsed.terms]
    notes = parsed.notes
    if len(terms) > MAX_TERMS:
        notes += f" (เอกสารมี {len(terms)} ภาคเรียน ระบบรับได้ {MAX_TERMS} ภาค จึงใช้ {MAX_TERMS} ภาคล่าสุด)"
        terms = terms[-MAX_TERMS:]

    credits = parsed.credits_earned if parsed.credits_earned is not None and 0 <= parsed.credits_earned <= 300 else None
    return {
        "student_id": parsed.student_id,
        "first_name": parsed.first_name,
        "last_name": parsed.last_name,
        "credits_earned": credits,
        "terms": terms,
        "notes": notes,
    }
