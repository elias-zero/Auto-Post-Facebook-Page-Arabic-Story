#!/usr/bin/env python3
# post_to_facebook_qwen.py  (مُحدّث لاستخدام Router API)
import os
import time
import requests
import sys
import json

# ---------- الإعدادات عبر GitHub Secrets ----------
HF_TOKEN = os.getenv("HF_API_TOKEN")
FB_PAGE_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")
FB_PAGE_ID = os.getenv("FB_PAGE_ID")

if not (HF_TOKEN and FB_PAGE_TOKEN and FB_PAGE_ID):
    print("ERROR: Missing environment variables. Set HF_API_TOKEN, FB_PAGE_ACCESS_TOKEN, FB_PAGE_ID as repository secrets.")
    sys.exit(1)

# --------- استخدام Router endpoint الجديد -------------
ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "Qwen/Qwen3-1.7B"  # أو "Qwen/Qwen3-1.7B:latest" إذا أردت تحديد الوسم
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

PROMPT_AR = (
    "أنت كاتب عربي محترف. اكتب مخرجاً متسقاً ومهيّأ بصيغة نصية قابلة للتحليل:\n"
    "- عنوان القصة (سطر واحد)\n"
    "- فاصل: ###STORY###\n"
    "- القصة نفسها (120-220 كلمة تقريباً، بالعربية الفصحى، لغة واضحة، قصة أخلاقية فيها عبرة)\n"
    "- فاصل: ###LESSON###\n"
    "- العبرة (سطر أو سطرين).\n\n"
    "شروط صارمة: لا محتوى جنسي/فاحش، لا دعوة لدين آخر، لا سب أو إهانات، أسلوب ملهم ومناسب للفيسبوك.\n"
    "أعِد الإخراج بنفس الفواصل المذكورة بالضبط."
)

FORBIDDEN_KEYWORDS = [
    "جنس", "أغراء", "مضاجعة", "ممارسة", "عاهرة", "عاهر", "شرم", "فرج", "عارية", "مفاتن",
    "لعنة", "غبي", "أحمق", "خرا", "قحبة",
    "مسيح", "نصراني", "مسيحية", "يهودي", "يهودية", "بوذي", "هندوسي", "شرك",
    "كفر", "الالحاد", "شتيمة لله", "سب الله"
]
LOWER_FORBIDDEN = [w.lower() for w in FORBIDDEN_KEYWORDS]

def generate_with_qwen_chat(prompt, max_retries=3):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        # معلمات شبيهة بالـ OpenAI
        "temperature": 0.8,
        "max_tokens": 360,
        "top_p": 0.95,
        "n": 1
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(ROUTER_CHAT_URL, headers=HEADERS, json=payload, timeout=120)
            if r.status_code == 429:
                wait = 10 * attempt
                print(f"Rate limited (429). Waiting {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            # نبحث بعناية عن نص الرد ضمن أشكال متعددة ممكنة
            # شكل شائع: {"choices":[{"message":{"role":"assistant","content":"..."}}, ...]}
            text = None
            if isinstance(data, dict):
                choices = data.get("choices")
                if isinstance(choices, list) and len(choices) > 0:
                    first = choices[0]
                    # حالات متعددة
                    if isinstance(first.get("message"), dict) and "content" in first["message"]:
                        text = first["message"]["content"]
                    elif "text" in first:
                        text = first["text"]
                    elif "message" in first and isinstance(first["message"], str):
                        text = first["message"]
                # بعض صيغ Router قد ترجع 'output' أو 'output_text'
                if not text:
                    if "output_text" in data:
                        text = data["output_text"]
                    elif "output" in data and isinstance(data["output"], str):
                        text = data["output"]
            # fallback: stringify
            if not text:
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

def parse_generated(text):
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
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 3:
        title = lines[0]
        lesson = lines[-1]
        story = " ".join(lines[1:-1])
        return title, story, lesson
    return None, None, None

def contains_forbidden(text):
    lowered = text.lower()
    for fw in LOWER_FORBIDDEN:
        if fw in lowered:
            return True, fw
    return False, None

def tidy_text(s, limit=3000):
    if len(s) <= limit:
        return s
    cut = s[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"

def post_to_facebook(message):
    graph_url = f"https://graph.facebook.com/v17.0/{FB_PAGE_ID}/feed"
    payload = {"message": message, "access_token": FB_PAGE_TOKEN}
    r = requests.post(graph_url, data=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def main():
    for attempt in range(1, 6):
        print(f"[Attempt {attempt}] Requesting generation from HF Router...")
        raw = generate_with_qwen_chat(PROMPT_AR)
        print("Raw generation preview:", raw[:400].replace("\n", " ") + ("..." if len(raw) > 400 else ""))
        title, story, lesson = parse_generated(raw)
        if not title or not story or not lesson:
            print("Parsing failed or incomplete structure, retrying...")
            time.sleep(2)
            continue
        combined = " ".join([title, story, lesson])
        bad, word = contains_forbidden(combined)
        if bad:
            print(f"Found forbidden keyword '{word}' in generation — retrying...")
            time.sleep(2)
            continue
        final_text = f"{title}\n\n{story}\n\nالعبرة: {lesson}\n\n#قصص\n\nإذا أعجبتك القصة لا تنسى متابعة الصفحة."
        final_text = tidy_text(final_text, limit=3000)
        try:
            print("Posting to Facebook (preview first 300 chars):", final_text[:300])
            res = post_to_facebook(final_text)
            print("Posted successfully. Response:", res)
            return
        except requests.HTTPError as e:
            print("Facebook HTTP error:", e, getattr(e.response, "text", ""))
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
