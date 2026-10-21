import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from app.api import predictions, document_reader
from app.core.database import engine, Base

# สร้างตารางใน Database อัตโนมัติ
try:
    Base.metadata.create_all(bind=engine)
    # create_all ไม่เพิ่มคอลัมน์ใหม่ให้ตารางที่มีอยู่แล้ว จึงต้องเพิ่ม academic_year เอง
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS academic_year INTEGER"))
except Exception as e:
    print("คำเตือนการเชื่อมต่อฐานข้อมูล:", e)

app = FastAPI(
    title="ระบบสนับสนุนการตัดสินใจทำนายความเสี่ยงพ้นสภาพนักศึกษา (DSS)",
    description="ระบบสำหรับอาจารย์ที่ปรึกษา เพื่อวิเคราะห์ทำนายความเสี่ยงของนักศึกษาด้วย AI และ SHAP",
    version="1.0.0"
)

# ลงทะเบียน API Router
app.include_router(predictions.router)
app.include_router(document_reader.router)

# กำหนดให้หน้าหลัก (/) โหลดไฟล์ index.html ที่สร้างไว้ขึ้นมาแสดงผล
@app.get("/", response_class=HTMLResponse, tags=["หน้าเว็บหลัก"])
def home_page():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ไม่พบไฟล์ index.html กรุณาตรวจสอบว่ามีไฟล์ index.html อยู่ที่โฟลเดอร์หลักของโปรเจกต์หรือยัง</h1>"