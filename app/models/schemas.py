from pydantic import BaseModel, Field
from typing import List

class ผลการเรียนรายภาค(BaseModel):
    gpax: float = Field(..., ge=0, le=4, title="เกรดเฉลี่ยสะสม (GPAX) ณ สิ้นภาคเรียน", example=1.75)
    fw_count: int = Field(..., ge=0, le=15, title="จำนวนวิชาที่ได้ F หรือ W ในภาคเรียนนี้", example=2)

class ข้อมูลนักศึกษาขาเข้า(BaseModel):
    student_id: str = Field(..., title="รหัสนักศึกษา", example="67123456")
    first_name: str = Field(..., title="ชื่อจริง", example="สมชาย")
    last_name: str = Field(..., title="นามสกุล", example="ใจดี")
    academic_year: int = Field(..., ge=1, le=4, title="ชั้นปี (1-4)", example=2)
    term_results: List[ผลการเรียนรายภาค] = Field(
        ..., min_length=1, max_length=8,
        title="ผลการเรียนรายภาคเรียน (เรียงจากภาคแรกถึงภาคล่าสุด 1-8 ภาค)",
        example=[{"gpax": 2.10, "fw_count": 1}, {"gpax": 1.85, "fw_count": 1}, {"gpax": 1.75, "fw_count": 2}]
    )
    credits_earned: int = Field(..., ge=0, le=300, title="หน่วยกิตที่สะสมได้", example=45)

    class Config:
        title = "ข้อมูลนักศึกษาสำหรับวิเคราะห์"

class ปัจจัยความเสี่ยงจากAI(BaseModel):
    feature_name: str = Field(..., title="ชื่อปัจจัย")
    current_value: float = Field(..., title="ค่าปัจจุบัน")
    risk_contribution: float = Field(..., title="ผลกระทบต่อความเสี่ยง")

    class Config:
        title = "ปัจจัยที่มีผลต่อความเสี่ยง (SHAP)"

class ผลการทำนายความเสี่ยง(BaseModel):
    student_id: str = Field(..., title="รหัสนักศึกษา")
    student_name: str = Field(..., title="ชื่อ-นามสกุล นักศึกษา")
    risk_score_percentage: float = Field(..., title="เปอร์เซ็นต์ความเสี่ยงพ้นสภาพ (%)")
    risk_status: str = Field(..., title="ระดับความเสี่ยง")
    key_factors: List[ปัจจัยความเสี่ยงจากAI] = Field(..., title="วิเคราะห์ปัจจัยสำคัญโดย AI")
    rule_alerts: List[str] = Field(default_factory=list, title="แจ้งเตือนตามเกณฑ์ข้อบังคับการพ้นสภาพ")

    class Config:
        title = "สรุปผลการวิเคราะห์ความเสี่ยง"
