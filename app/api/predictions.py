import pickle
import re
import pandas as pd
import shap
import traceback
import io
from typing import List
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.models.schemas import ข้อมูลนักศึกษาขาเข้า, ผลการทำนายความเสี่ยง, ปัจจัยความเสี่ยงจากAI
from app.core.database import get_db
from app.models.database_models import Student, RiskPrediction

router = APIRouter(prefix="/api/v1", tags=["ระบบจัดการและทำนายความเสี่ยงนักศึกษา"])

# โหลดโมเดล AI
model_path = "ml_models/xgb_dropout_model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
except Exception as e:
    model = None
    explainer = None
    print("⚠️ ไม่สามารถโหลดโมเดล AI ได้:", e)

# เกณฑ์ตามข้อบังคับการพ้นสภาพ (ค่าทั่วไปของมหาวิทยาลัยไทย ควรตรวจสอบกับข้อบังคับของมหาวิทยาลัยก่อนใช้งานจริง)
GPAX_DISMISS = 1.50          # GPAX ต่ำกว่านี้ = พ้นสภาพ
GPAX_MIN_GRADUATE = 2.00     # GPAX ต่ำกว่านี้ = ติดวิทยาทัณฑ์ / ยังจบการศึกษาไม่ได้
MAX_PROBATION_TERMS = 2      # ติดวิทยาทัณฑ์ติดต่อกันได้ไม่เกินกี่ภาคเรียน

# หน่วยกิตตามแผนการเรียนต่อภาคเรียนปกติ (ใช้คำนวณว่าเรียนได้ตามแผนหรือล่าช้า)
# ปรับให้ตรงกับหลักสูตร เช่น หลักสูตร 4 ปี 144 หน่วยกิต / 8 ภาค = 18
CREDITS_PER_TERM = 18

# เกณฑ์ระดับความเสี่ยงจากคะแนน AI: สีแดง 51 ขึ้นไป | สีเหลือง 41 - 50 | สีเขียว 40 ลงไป
RISK_LEVELS = [
    "🟢 กลุ่มปกติ (ไม่มีความเสี่ยง)",
    "🟡 กลุ่มเสี่ยงปานกลาง (ควรเฝ้าระวัง)",
    "🔴 กลุ่มเสี่ยงสูง (ต้องรีบดูแลด่วน)",
]

thai_names = {
    'gpax': 'เกรดเฉลี่ยสะสม (GPAX)',
    'credit_completion_rate': 'หน่วยกิตสะสมเทียบแผนการเรียน (%)',
    'fw_count': 'จำนวนวิชา F/W เทอมล่าสุด',
    'probation_terms': 'จำนวนภาคเรียนที่ติดวิทยาทัณฑ์',
}

MAX_TERMS = 8

# คอลัมน์ของไฟล์ Excel แบบเทมเพลต (ตรงกับข้อมูลที่กรอกในฟอร์ม)
# ผลการเรียนรายภาคใช้คอลัมน์ term1_gpax, term1_fw ... term8_gpax, term8_fw (ต้องมีอย่างน้อยภาคที่ 1)
TEMPLATE_COLUMNS = ['student_id', 'first_name', 'last_name', 'academic_year', 'credits_earned']
TERM_COLUMNS = [col for t in range(1, MAX_TERMS + 1) for col in (f'term{t}_gpax', f'term{t}_fw')]


def นับภาคเรียนวิทยาทัณฑ์(term_results):
    """นับจำนวนภาคเรียนล่าสุดที่ GPAX ต่ำกว่า 2.00 ติดต่อกัน (นับรวมภาคล่าสุด)"""
    count = 0
    for term in reversed(term_results):
        if term['gpax'] < GPAX_MIN_GRADUATE:
            count += 1
        else:
            break
    return count


def สร้างฟีเจอร์(term_results, credits_earned):
    """term_results: list ของ {'gpax', 'fw_count'} เรียงจากภาคแรกถึงภาคล่าสุด
    ใช้ GPAX และวิชา F/W ของภาคล่าสุด"""
    latest = term_results[-1]
    # หน่วยกิตตามแผน = จำนวนภาคที่เรียนแล้ว x หน่วยกิตต่อภาค
    # หน่วยกิตที่เรียนเร็วกว่าแผนมากไม่ได้ช่วยลดความเสี่ยงเพิ่ม จึงจำกัดไว้ที่ 120%
    credits_planned = len(term_results) * CREDITS_PER_TERM
    credit_rate = min(credits_earned / credits_planned * 100, 120.0)
    return {
        'gpax': float(latest['gpax']),
        'credit_completion_rate': round(credit_rate, 2),
        'fw_count': int(latest['fw_count']),
        'probation_terms': นับภาคเรียนวิทยาทัณฑ์(term_results),
    }


def ตรวจเกณฑ์ข้อบังคับ(gpax, probation_terms, academic_year, n_terms):
    """คืนค่า (ระดับความเสี่ยงขั้นต่ำตามข้อบังคับ 0/1/2, รายการแจ้งเตือน)"""
    alerts = []
    level = 0
    if gpax < GPAX_DISMISS:
        level = 2
        if n_terms >= 2:
            alerts.append(f"GPAX {gpax:.2f} ต่ำกว่า {GPAX_DISMISS:.2f} หลังเรียนครบ {n_terms} ภาคเรียน เข้าเกณฑ์พ้นสภาพนักศึกษา")
        else:
            # เกณฑ์ 1.50 มักเริ่มใช้หลังผ่าน 2 ภาคเรียนแรก
            alerts.append(f"GPAX {gpax:.2f} ต่ำกว่า {GPAX_DISMISS:.2f} ถ้าสิ้นภาคเรียนที่ 2 ยังต่ำกว่าเกณฑ์จะพ้นสภาพนักศึกษา")
    elif gpax < GPAX_MIN_GRADUATE:
        if probation_terms > MAX_PROBATION_TERMS:
            level = 2
            alerts.append(
                f"GPAX ต่ำกว่า {GPAX_MIN_GRADUATE:.2f} ติดต่อกัน {probation_terms} ภาคเรียน "
                f"(เกินเกณฑ์วิทยาทัณฑ์ {MAX_PROBATION_TERMS} ภาคเรียน) เข้าเกณฑ์พ้นสภาพนักศึกษา"
            )
        else:
            level = 1
            alerts.append(
                f"GPAX {gpax:.2f} อยู่ในช่วงวิทยาทัณฑ์ ({GPAX_DISMISS:.2f} - 1.99) ต่ำกว่า {GPAX_MIN_GRADUATE:.2f} "
                f"ติดต่อกัน {probation_terms} ภาคเรียน (พ้นสภาพเมื่อเกิน {MAX_PROBATION_TERMS} ภาคเรียน)"
            )
    if academic_year == 4 and gpax < GPAX_MIN_GRADUATE:
        alerts.append(f"ชั้นปี 4 แต่ GPAX ต่ำกว่า {GPAX_MIN_GRADUATE:.2f} ยังไม่สามารถสำเร็จการศึกษาได้")
    return level, alerts


def ประเมินความเสี่ยง(features, academic_year, n_terms):
    """ทำนายด้วย AI แล้วปรับระดับความเสี่ยงขึ้นตามเกณฑ์ข้อบังคับ (ใช้ระดับที่สูงกว่า)"""
    input_df = pd.DataFrame([features])[list(model.feature_names_in_)]

    prob = float(model.predict_proba(input_df)[0][1])
    score_percent = round(prob * 100, 2)
    if score_percent >= 51.0:
        ai_level = 2
    elif score_percent >= 41.0:
        ai_level = 1
    else:
        ai_level = 0

    rule_level, alerts = ตรวจเกณฑ์ข้อบังคับ(features['gpax'], features['probation_terms'], academic_year, n_terms)
    if rule_level > ai_level:
        alerts.append("ระดับความเสี่ยงถูกปรับขึ้นตามเกณฑ์ข้อบังคับการพ้นสภาพ")
    risk_status = RISK_LEVELS[max(ai_level, rule_level)]

    shap_values = explainer(input_df)
    key_factors = []
    for name, val, shap_val in zip(input_df.columns, input_df.iloc[0], shap_values.values[0]):
        if hasattr(shap_val, "__len__"):
            shap_val = float(shap_val[1]) if len(shap_val) > 1 else float(shap_val[0])
        else:
            shap_val = float(shap_val)
        key_factors.append(
            ปัจจัยความเสี่ยงจากAI(
                feature_name=thai_names.get(name, name),
                current_value=float(val),
                risk_contribution=shap_val
            )
        )
    key_factors.sort(key=lambda x: abs(x.risk_contribution), reverse=True)

    return score_percent, risk_status, key_factors, alerts


def บันทึกผลการประเมิน(db, student_id, first_name, last_name, academic_year,
                      term_results, credits_earned):
    features = สร้างฟีเจอร์(term_results, credits_earned)
    db_student = db.query(Student).filter(Student.student_id == student_id).first()
    if not db_student:
        db_student = Student(
            student_id=student_id,
            first_name=first_name,
            last_name=last_name,
            highschool_gpa=features['gpax'],
            academic_year=academic_year
        )
        db.add(db_student)
    else:
        db_student.highschool_gpa = features['gpax']
        if academic_year is not None:
            db_student.academic_year = academic_year
    db.commit()

    score_percent, risk_status, key_factors, alerts = ประเมินความเสี่ยง(features, academic_year, len(term_results))

    db_prediction = RiskPrediction(
        student_id=student_id,
        risk_score=score_percent,
        risk_level=risk_status,
        shap_explanation={
            "ปัจจัยหลัก": [f.dict() for f in key_factors],
            "ข้อมูลที่ใช้ทำนาย": features,
            "ผลการเรียนรายภาค": term_results,
            "เกณฑ์ข้อบังคับ": alerts,
        }
    )
    db.add(db_prediction)
    db.commit()
    return score_percent, risk_status, key_factors, alerts


@router.post(
    "/predict-student",
    response_model=ผลการทำนายความเสี่ยง,
    summary="ทำนายความเสี่ยงและบันทึกข้อมูลนักศึกษา"
)
def ทำนายและบันทึกข้อมูลนักศึกษา(data: ข้อมูลนักศึกษาขาเข้า, db: Session = Depends(get_db)):
    if model is None or explainer is None:
        raise HTTPException(status_code=500, detail="ยังไม่ได้โหลดโมเดล AI ในระบบ")

    try:
        score_percent, risk_status, key_factors, alerts = บันทึกผลการประเมิน(
            db, data.student_id, data.first_name, data.last_name, data.academic_year,
            [t.model_dump() for t in data.term_results],
            data.credits_earned
        )

        return ผลการทำนายความเสี่ยง(
            student_id=data.student_id,
            student_name=f"{data.first_name} {data.last_name}",
            risk_score_percentage=score_percent,
            risk_status=risk_status,
            key_factors=key_factors,
            rule_alerts=alerts
        )
    except Exception as e:
        db.rollback()
        print("❌ เกิดข้อผิดพลาดขณะทำนายผล:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/students", summary="ดึงรายชื่อนักศึกษาทั้งหมด")
def ดึงรายชื่อนักศึกษา(db: Session = Depends(get_db)):
    students = db.query(Student).all()
    results = []
    for s in students:
        latest_pred = db.query(RiskPrediction).filter(RiskPrediction.student_id == s.student_id).order_by(RiskPrediction.predicted_at.desc()).first()
        results.append({
            "student_id": s.student_id,
            "name": f"{s.first_name} {s.last_name}",
            "academic_year": s.academic_year,
            "risk_score": latest_pred.risk_score if latest_pred else 0,
            "risk_status": latest_pred.risk_level if latest_pred else "ยังไม่ได้ประเมิน"
        })
    return results

@router.delete("/delete-student/{student_id}", summary="ลบนักศึกษา")
def ลบนักศึกษา(student_id: str, db: Session = Depends(get_db)):
    db.query(RiskPrediction).filter(RiskPrediction.student_id == student_id).delete()
    db.query(Student).filter(Student.student_id == student_id).delete()
    db.commit()
    return {"message": "ลบข้อมูลสำเร็จ"}

@router.delete("/delete-all-students", summary="ลบข้อมูลนักศึกษาและประวัติการทำนายทั้งหมด")
def ลบข้อมูลนักศึกษาทั้งหมด(db: Session = Depends(get_db)):
    try:
        db.query(RiskPrediction).delete()
        db.query(Student).delete()
        db.commit()
        return {"message": "ลบข้อมูลนักศึกษาและประวัติทั้งหมดในระบบเรียบร้อยแล้ว"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export-excel", summary="ดาวน์โหลดรายชื่อนักศึกษาทั้งหมดเป็นไฟล์ Excel")
def ดาวน์โหลดไฟล์Excel(db: Session = Depends(get_db)):
    students = db.query(Student).all()
    data_list = []

    for s in students:
        latest_pred = db.query(RiskPrediction).filter(RiskPrediction.student_id == s.student_id).order_by(RiskPrediction.predicted_at.desc()).first()
        explanation = (latest_pred.shap_explanation or {}) if latest_pred else {}
        data_list.append({
            "รหัสนักศึกษา": s.student_id,
            "ชื่อ-นามสกุล": f"{s.first_name} {s.last_name}",
            "ชั้นปี": s.academic_year if s.academic_year and 1 <= s.academic_year <= 4 else "-",
            "เกรดเฉลี่ย (GPAX)": s.highschool_gpa,
            "คะแนนความเสี่ยง (%)": latest_pred.risk_score if latest_pred else 0,
            "ระดับความเสี่ยง": latest_pred.risk_level if latest_pred else "ยังไม่ได้ประเมิน",
            "แจ้งเตือนตามข้อบังคับ": " | ".join(explanation.get("เกณฑ์ข้อบังคับ", [])) or "-",
            "วันที่ประเมินล่าสุด": str(latest_pred.predicted_at) if latest_pred else "-"
        })

    if not data_list:
        data_list = [{"รหัสนักศึกษา": "-", "ชื่อ-นามสกุล": "-", "ชั้นปี": "-", "เกรดเฉลี่ย (GPAX)": "-", "คะแนนความเสี่ยง (%)": "-", "ระดับความเสี่ยง": "-", "แจ้งเตือนตามข้อบังคับ": "-", "วันที่ประเมินล่าสุด": "-"}]

    df = pd.DataFrame(data_list)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='รายงานความเสี่ยงนักศึกษา')
    output.seek(0)

    headers = {'Content-Disposition': 'attachment; filename="student_risk_report.xlsx"'}
    return StreamingResponse(output, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers=headers)

@router.get("/import-template", summary="ดาวน์โหลดไฟล์ Excel ตัวอย่างสำหรับนำเข้าข้อมูล")
def ดาวน์โหลดเทมเพลตExcel():
    # ภาคเรียนที่ยังไม่ได้เรียนให้เว้นว่างไว้
    df = pd.DataFrame([
        {"student_id": "67123456", "first_name": "สมชาย", "last_name": "ใจดี", "academic_year": 2,
         "credits_earned": 45,
         "term1_gpax": 2.10, "term1_fw": 1, "term2_gpax": 1.85, "term2_fw": 1, "term3_gpax": 1.75, "term3_fw": 2},
        {"student_id": "67123457", "first_name": "สมหญิง", "last_name": "ตั้งใจ", "academic_year": 1,
         "credits_earned": 18,
         "term1_gpax": 3.20, "term1_fw": 0},
    ], columns=TEMPLATE_COLUMNS + TERM_COLUMNS)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='ข้อมูลนักศึกษา')
    output.seek(0)
    headers = {'Content-Disposition': 'attachment; filename="student_import_template.xlsx"'}
    return StreamingResponse(output, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers=headers)

# ===== อ่านไฟล์ Excel/CSV ที่หัวคอลัมน์เป็นภาษาไทยหรืออังกฤษก็ได้ =====
# ชื่อหัวคอลัมน์ที่รองรับ (เทียบหลังตัดช่องว่าง/สัญลักษณ์ และแปลงเป็นตัวพิมพ์เล็ก)
COLUMN_ALIASES = {
    'student_id': ['studentid', 'id', 'รหัส', 'รหัสนักศึกษา', 'รหัสนศ', 'รหัสประจำตัว', 'รหัสประจำตัวนักศึกษา'],
    'first_name': ['firstname', 'ชื่อ', 'ชื่อจริง'],
    'last_name': ['lastname', 'surname', 'นามสกุล'],
    'full_name': ['fullname', 'name', 'ชื่อนามสกุล', 'ชื่อสกุล'],
    'academic_year': ['academicyear', 'year', 'yearlevel', 'ชั้นปี', 'ปี', 'ชั้นปีที่'],
    'credits_earned': ['creditsearned', 'credits', 'credit', 'หน่วยกิต', 'หน่วยกิตสะสม', 'หน่วยกิตที่สะสมได้',
                       'หน่วยกิตที่ได้', 'หน่วยกิตที่สอบผ่าน'],
}
GPAX_WORDS = ('gpax', 'gpa', 'เกรดเฉลี่ย', 'เกรด')
FW_WORDS = ('fw', 'ติดf', 'วิชาf', 'จำนวนf')


def ทำให้เป็นมาตรฐาน(text):
    return re.sub(r'[\s_\-\.\(\)\[\]/:]+', '', str(text)).lower()


def แปลงหัวคอลัมน์(header_row):
    """คืนค่า dict ชื่อฟิลด์ -> ลำดับคอลัมน์ และ term_cols {ภาค: {'gpax': col, 'fw': col}}"""
    fields, term_cols, single = {}, {}, {}
    for col, raw in enumerate(header_row):
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        key = ทำให้เป็นมาตรฐาน(raw)
        if not key:
            continue
        kind = 'gpax' if any(w in key for w in GPAX_WORDS) else 'fw' if any(w in key for w in FW_WORDS) else None
        if kind:
            # เลขภาค 1-8 ที่อยู่ในหัวคอลัมน์ (ไม่นับเลขยาว เช่น ปีการศึกษา 2567)
            terms = [int(n) for n in re.findall(r'\d+', key) if len(n) == 1 and 1 <= int(n) <= MAX_TERMS]
            if terms:
                term_cols.setdefault(terms[0], {})[kind] = col
            else:
                single.setdefault(kind, col)
            continue
        for field, aliases in COLUMN_ALIASES.items():
            if key in aliases and field not in fields:
                fields[field] = col
                break
    # ไฟล์ที่มี GPAX คอลัมน์เดียว (ไม่แยกภาค) ให้ถือเป็นผลการเรียนภาคล่าสุด 1 ภาค
    if not term_cols and 'gpax' in single:
        term_cols[1] = single
    return fields, term_cols


def ค่าว่าง(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ('', 'nan', '-')


def ข้อความรหัส(v):
    s = str(v).strip()
    return s[:-2] if s.endswith('.0') else s


def อ่านแถวนักศึกษา(row, fields, term_cols):
    """คืนค่า (ข้อมูลสำหรับบันทึก, None) หรือ (None, เหตุผลที่ข้าม)"""
    sid = row[fields['student_id']] if 'student_id' in fields else None
    if ค่าว่าง(sid):
        return None, None  # แถวว่าง ไม่นับเป็นแถวผิดพลาด
    student_id = ข้อความรหัส(sid)

    if 'first_name' in fields:
        first_name = '' if ค่าว่าง(row[fields['first_name']]) else str(row[fields['first_name']]).strip()
        last_name = '-' if 'last_name' not in fields or ค่าว่าง(row[fields['last_name']]) else str(row[fields['last_name']]).strip()
    elif 'full_name' in fields and not ค่าว่าง(row[fields['full_name']]):
        parts = str(row[fields['full_name']]).strip().split(None, 1)
        first_name, last_name = parts[0], parts[1] if len(parts) > 1 else '-'
    else:
        first_name, last_name = '-', '-'

    term_results = []
    try:
        for t in sorted(term_cols):
            gpax_val = row[term_cols[t]['gpax']] if 'gpax' in term_cols[t] else None
            if ค่าว่าง(gpax_val):
                break  # อ่านต่อกันจนเจอภาคที่เว้นว่าง
            fw_col = term_cols[t].get('fw')
            fw_val = row[fw_col] if fw_col is not None else None
            term_results.append({'gpax': float(gpax_val), 'fw_count': 0 if ค่าว่าง(fw_val) else int(float(fw_val))})
    except (TypeError, ValueError):
        return None, f"{student_id}: GPAX หรือจำนวนวิชา F/W ไม่ใช่ตัวเลข"
    if not term_results:
        return None, f"{student_id}: ไม่มี GPAX"
    if not all(0 <= t['gpax'] <= 4 and t['fw_count'] >= 0 for t in term_results):
        return None, f"{student_id}: GPAX ต้องอยู่ระหว่าง 0-4 และ F/W ต้องไม่ติดลบ"

    academic_year = None
    if 'academic_year' in fields and not ค่าว่าง(row[fields['academic_year']]):
        try:
            academic_year = int(float(row[fields['academic_year']]))
        except (TypeError, ValueError):
            academic_year = None
        if academic_year is not None and not 1 <= academic_year <= 4:
            return None, f"{student_id}: ชั้นปีต้องเป็น 1-4"

    # ไม่มีหน่วยกิต -> ถือว่าเรียนได้ตามแผน
    credits_default = len(term_results) * CREDITS_PER_TERM
    credits_earned = credits_default
    if 'credits_earned' in fields and not ค่าว่าง(row[fields['credits_earned']]):
        try:
            credits_earned = int(float(row[fields['credits_earned']]))
        except (TypeError, ValueError):
            return None, f"{student_id}: หน่วยกิตไม่ใช่ตัวเลข"
        if credits_earned < 0:
            return None, f"{student_id}: หน่วยกิตติดลบ"

    return (student_id, first_name or '-', last_name, academic_year, term_results, credits_earned), None


def อ่านตารางจากไฟล์(contents, filename):
    name = (filename or '').lower()
    if name.endswith('.csv'):
        for enc in ('utf-8-sig', 'cp874'):
            try:
                return [pd.read_csv(io.BytesIO(contents), header=None, encoding=enc, dtype=object)]
            except UnicodeDecodeError:
                continue
        raise HTTPException(status_code=400, detail="อ่านไฟล์ CSV ไม่ได้ กรุณาบันทึกเป็น UTF-8 หรือใช้ไฟล์ Excel")
    try:
        sheets = pd.read_excel(io.BytesIO(contents), sheet_name=None, header=None, dtype=object)
    except Exception:
        raise HTTPException(status_code=400, detail="เปิดไฟล์ไม่ได้ กรุณาใช้ไฟล์ Excel (.xlsx, .xls) หรือ .csv")
    return list(sheets.values())


@router.post("/import-excel", summary="อัปโหลดไฟล์ Excel/CSV เพื่อนำเข้าและทำนายผลนักศึกษาหลายคน")
def นำเข้าไฟล์Excel(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if model is None or explainer is None:
        raise HTTPException(status_code=500, detail="ยังไม่ได้โหลดโมเดล AI ในระบบ")

    contents = file.file.read()
    tables = อ่านตารางจากไฟล์(contents, file.filename)

    # หาแถวหัวคอลัมน์ในทุกชีต (อาจมีชื่อรายงานอยู่บรรทัดบน ๆ) ต้องมีรหัสนักศึกษาและ GPAX
    found = None
    seen_headers = []
    for df in tables:
        for r in range(min(15, len(df))):
            fields, term_cols = แปลงหัวคอลัมน์(df.iloc[r].tolist())
            if 'student_id' in fields and any('gpax' in c for c in term_cols.values()):
                found = (df, r, fields, term_cols)
                break
            if r == 0:
                seen_headers = [str(v) for v in df.iloc[r].tolist() if not ค่าว่าง(v)]
        if found:
            break

    if not found:
        raise HTTPException(status_code=400, detail=(
            "ไม่พบคอลัมน์ที่ระบบต้องใช้ ต้องมีอย่างน้อย: รหัสนักศึกษา และ GPAX "
            "(เช่น 'GPAX ภาค 1', 'GPAX ภาค 2' หรือ term1_gpax) แนะนำให้กดโหลดไฟล์ตัวอย่าง (Template) แล้วกรอกตามนั้น"
            + (f" | หัวคอลัมน์ในไฟล์: {', '.join(seen_headers[:12])}" if seen_headers else "")
        ))

    df, header_row, fields, term_cols = found
    imported_count = 0
    skipped = []
    try:
        for r in range(header_row + 1, len(df)):
            parsed, reason = อ่านแถวนักศึกษา(df.iloc[r].tolist(), fields, term_cols)
            if reason:
                skipped.append(reason)
            if parsed:
                บันทึกผลการประเมิน(db, *parsed)
                imported_count += 1
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    if imported_count == 0:
        raise HTTPException(status_code=400, detail="ไม่มีนักศึกษาที่นำเข้าได้" + (
            f" | ตัวอย่างปัญหา: {'; '.join(skipped[:5])}" if skipped else " (ไม่มีข้อมูลใต้หัวคอลัมน์)"))

    message = f"นำเข้าและประเมินผลสำเร็จ {imported_count} คน"
    missing = [label for key, label in [('credits_earned', 'หน่วยกิต (ถือว่าเรียนตามแผน)'),
                                        ('academic_year', 'ชั้นปี')] if key not in fields]
    if missing:
        message += f"\nไฟล์ไม่มีคอลัมน์: {', '.join(missing)}"
    if skipped:
        message += f"\nข้าม {len(skipped)} แถว: {'; '.join(skipped[:5])}" + (" ..." if len(skipped) > 5 else "")
    return {"message": message}
