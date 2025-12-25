#!/usr/bin/env python3
# post_to_facebook_qwen.py
import os
import time
import requests
import sys
import json
import re

# ---------- الإعدادات عبر GitHub Secrets ----------
HF_TOKEN = os.getenv("HF_API_TOKEN")
FB_PAGE_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")
FB_PAGE_ID = os.getenv("FB_PAGE_ID")

if not (HF_TOKEN and FB_PAGE_TOKEN and FB_PAGE_ID):
    print("ERROR: Missing environment variables. Set HF_API_TOKEN, FB_PAGE_ACCESS_TOKEN, FB_PAGE_ID as repository secrets.")
    sys.exit(1)

MODEL = "Qwen/Qwen3-1.7B"
HF_ENDPOINT = f"https://api-inference.huggingface.co/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

# ---------- إعداد الـ prompt لطلب إخراج منسق قابل للمعالجة ----------
PROMPT_AR = (
    "أنت كاتب عربي محترف. اكتب مخرجاً متسقاً ومهيئاً بصيغة نصية قابلة للتحليل:"
    "\n- عنوان القصة (سطر واحد)"
    "\n- فاصل: ###STORY###"
    "\n- القصة نفسها (120-220 كلمة تقريباً، بالعربية الفصحى، لغة واضحة، قصة أخلاقية فيها عبرة)"
    "\n- فاصل: ###LESSON###"
    "\n- العبرة (سطر أو سطرين)."
    "\n\nشروط صارمة للإخراج:"
    "\n1) لا تتضمن أي محتوى جنسي أو فاحش أو إباحي."
    "\n2) لا تدعو إلى دين آخر أو تروج له؛ وتجنب أي خطاب تحريضي ديني أو كفري."
    "\n3) لا تستخدم ألفاظ سب أو سبّ الناس أو عنصرية."
    "\n4) استخدم أسلوب ملهم وهادئ مناسب للنشر العام على صفحة فيسبوك."
    "\n5) لا تتعدى القصة الحدود الأخلاقية؛ الهدف إعطاء عبرة أخلاقية عامة."
    "\n\nأعطِ النص النهائي بشكلٍ نصي يحتوي العناوين والفواصل كما طُلب بالضبط."
)

# ---------- كلمات/مصطلحات محظورة بسيطة (قائمة أولية) ----------
FORBIDDEN_KEYWORDS = [
    # محتوى جنسي/فاحش
    "جنس", "أغراء", "مضاجعة", "ممارسة", "عاهرة", "عاهر", "شرم", "فرج", "عارية", "مفاتن",
    # سب/إهانات عامة
    "لعنة", "غبي", "أحمق", "خرا", "قحبة",
    # إشارات لأديان أخرى - نمنع ذكر دعوات أو ترويج
    "مسيح", "نصراني", "مسيحية", "يهودي", "يهودية", "بوذي", "هندوسي", "شرك",
    # كفر/إلحاد صريح أو ألفاظ مسيئة دينية
    "كفر", "الالحاد", "شتيمة لله", "سب الله"
]

# قائمة إضافية للفلترة - يمكن تعديلها لاحقا
LOWER_FORBIDDEN = [w.lower() for w in FORBIDDEN_KEYWORDS]

# ---------- توليد النص من HF مع محاولات إعادة عند فشل ----------
def generate_with_qwen(prompt, max_retries=3):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 360,
            "temperature": 0.8,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(HF_ENDPOINT, headers=HEADERS, json=payload, timeout=60)
            if r.status_code == 429:
                wait = 10 * attempt
                print(f"Rate limited (429). Waiting {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            # HF قد تعيد أشياء مختلفة حسب النموذج
            if isinstance(data, list) and data and "generated_text" in data[0]:
                text = data[0]["generated_text"]
            elif isinstance(data, dict) and "generated_text" in data:
                text = data["generated_text"]
            elif isinstance(data, dict) and "choices" in data and isinstance(data["choices"], list):
                text = data["choices"][0].get("text", "")
            else:
                # fallback: stringify
                text = json.dumps(data, ensure_ascii=False)
            return text.strip()
        except requests.HTTPError as e:
            print("HTTPError:", e, getattr(e.response, "text", ""))
            if attempt < max_retries:
                time.sleep(5 * attempt)
                continue
            raise
        except Exception as e:
            print("Error:", e)
            if attempt < max_retries:
                time.sleep(3 * attempt)
                continue
            raise
    raise RuntimeError("Failed to get generation after retries")

# ---------- تحليل النص المولد لاستخلاص العنوان/القصة/العبرة ----------
def parse_generated(text):
    # نبحث عن الفواصل المحددة ###STORY### و ###LESSON###
    # أحياناً يعود النص مع الحقول مضمّنة أو بدون فواصل؛ نحاول تغطية الحالتين
    if "###STORY###" in text and "###LESSON###" in text:
        try:
            title_part, rest = text.split("###STORY###", 1)
            story_part, lesson_part = rest.split("###LESSON###", 1)
            title = title_part.strip().splitlines()[0].strip()
            story = story_part.strip()
            lesson = lesson_part.strip().splitlines()[0].strip()
            return title, story, lesson
        except Exception:
            pass
    # fallback بسيط: حاول إيجاد أول سطر كعنوان، وبعده فقرة للقصة ثم السطر الأخير كعبرة
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 3:
        title = lines[0]
        lesson = lines[-1]
        story = " ".join(lines[1:-1])
        return title, story, lesson
    # لم نستطع التحليل بشكل جيد
    return None, None, None

# ---------- فحص المحتوى ضد قائمة المحظورات ----------
def contains_forbidden(text):
    lowered = text.lower()
    for fw in LOWER_FORBIDDEN:
        if fw in lowered:
            return True, fw
    return False, None

# ---------- تنظيف/تقليم النص قبل النشر ----------
def tidy_text(s, limit=2500):
    if len(s) <= limit:
        return s
    cut = s[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"

# ---------- نشر على فيسبوك ----------
def post_to_facebook(message):
    graph_url = f"https://graph.facebook.com/v17.0/{FB_PAGE_ID}/feed"
    payload = {"message": message, "access_token": FB_PAGE_TOKEN}
    r = requests.post(graph_url, data=payload, timeout=20)
    r.raise_for_status()
    return r.json()

# ---------- التنفيذ الرئيسي ----------
def main():
    for attempt in range(1, 6):  # نحاول حتى 5 مرات لاستلام نص مناسب
        print(f"[Attempt {attempt}] Requesting generation from HF...")
        raw = generate_with_qwen(PROMPT_AR)
        print("Raw generation preview:", raw[:400].replace("\n", " ") + ("..." if len(raw) > 400 else ""))
        title, story, lesson = parse_generated(raw)
        if not title or not story or not lesson:
            print("Parsing failed or incomplete structure, retrying...")
            time.sleep(2)
            continue

        # نتحقق من الكلمات المحظورة في كل جزء
        combined = " ".join([title, story, lesson])
        bad, word = contains_forbidden(combined)
        if bad:
            print(f"Found forbidden keyword '{word}' in generation — retrying...")
            time.sleep(2)
            continue

        # بناء النص النهائي حسب القالب المطلوب
        final_text = f"{title}\n\n{story}\n\nالعبرة: {lesson}\n\n#قصص\n\nإذا أعجبتك القصة لا تنسى متابعة الصفحة."
        final_text = tidy_text(final_text, limit=3000)  # حد عملي

        # محاولة النشر
        try:
            print("Posting to Facebook (preview first 300 chars):", final_text[:300])
            res = post_to_facebook(final_text)
            print("Posted successfully. Response:", res)
            return
        except requests.HTTPError as e:
            print("Facebook HTTP error:", e, getattr(e.response, "text", ""))
            # إذا كان خطأ مؤقت نجرب إعادة المحاولة
            if attempt < 5:
                time.sleep(5 * attempt)
                continue
            else:
                raise
        except Exception as e:
            print("Error posting to Facebook:", e)
            if attempt < 5:
                time.sleep(5 * attempt)
                continue
            else:
                raise

    print("Failed to generate a suitable story after multiple attempts.")
    sys.exit(1)

if __name__ == "__main__":
    main()
