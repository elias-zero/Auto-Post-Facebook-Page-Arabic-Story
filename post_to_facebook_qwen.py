#!/usr/bin/env python3
# post_to_facebook_qwen.py (مرن: يقرأ HF_MODEL من env، يطبع استجابة Router عند الخطأ)

import os, time, requests, sys, json

HF_TOKEN = os.getenv("HF_API_TOKEN")
FB_PAGE_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")
FB_PAGE_ID = os.getenv("FB_PAGE_ID")
HF_MODEL = os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-V3.2")  # الافتراضي الآن deepseek

if not (HF_TOKEN and FB_PAGE_TOKEN and FB_PAGE_ID):
    print("ERROR: Missing HF_API_TOKEN, FB_PAGE_ACCESS_TOKEN or FB_PAGE_ID.")
    sys.exit(1)

ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

PROMPT_AR = (
    "أنت كاتب عربي محترف. اكتب مخرجاً متسقاً بصيغة قابلة للتحليل:\n"
    "- عنوان القصة (سطر واحد)\n"
    "- فاصل: ###STORY###\n"
    "- القصة (120-220 كلمة بالعربية الفصحى)\n"
    "- فاصل: ###LESSON###\n"
    "- العبرة (سطر أو سطرين).\n"
    "شروط: لا محتوى جنسي أو إساءة أو دعوة لدين غير الإسلام. أنت مُلزم بالالتزام."
)

FORBIDDEN = [
    "جنس","أغراء","مضاجعة","ممارسة","عاهرة","عاهر","شرم","فرج","عارية","مفاتن",
    "لعنة","غبي","أحمق","خرا","قحبة",
    "مسيح","نصراني","مسيحية","يهودي","يهودية","بوذي","هندوسي","شرك",
    "كفر","الالحاد","شتيمة لله","سب الله"
]
LOWER_FORB = [w.lower() for w in FORBIDDEN]

def generate_with_router(prompt, model, retries=3):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 360,
        "top_p": 0.95,
        "n": 1
    }
    for attempt in range(1, retries+1):
        r = requests.post(ROUTER_CHAT_URL, headers=HEADERS, json=payload, timeout=120)
        if r.status_code == 429:
            wait = 10 * attempt
            print(f"Rate limited, sleeping {wait}s (attempt {attempt})")
            time.sleep(wait)
            continue
        if r.status_code == 404:
            print("404 from Router. Response JSON (diagnostic):")
            try:
                print(json.dumps(r.json(), ensure_ascii=False, indent=2))
            except Exception:
                print(r.text)
            r.raise_for_status()
        try:
            r.raise_for_status()
            data = r.json()
            # استخراج النص من الصيغ الشائعة
            text = None
            if isinstance(data, dict):
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first.get("message"), dict) and "content" in first["message"]:
                        text = first["message"]["content"]
                    elif "text" in first:
                        text = first["text"]
                if not text:
                    text = data.get("output_text") or data.get("output")
            if not text:
                text = json.dumps(data, ensure_ascii=False)
            return text.strip()
        except requests.HTTPError as e:
            print("HTTP error on attempt", attempt, "-", e)
            if attempt < retries:
                time.sleep(5 * attempt)
                continue
            raise
    raise RuntimeError("Failed to get model output")

def parse_generated(text):
    if "###STORY###" in text and "###LESSON###" in text:
        try:
            title_part, rest = text.split("###STORY###",1)
            story_part, lesson_part = rest.split("###LESSON###",1)
            title = title_part.strip().splitlines()[0].strip()
            story = story_part.strip()
            lesson = lesson_part.strip().splitlines()[0].strip()
            return title, story, lesson
        except:
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
    for fw in LOWER_FORB:
        if fw in lowered:
            return True, fw
    return False, None

def tidy_text(s, limit=3000):
    if len(s) <= limit:
        return s
    cut = s[:limit]
    if " " in cut:
        cut = cut.rsplit(" ",1)[0]
    return cut + "…"

def post_to_facebook(message):
    url = f"https://graph.facebook.com/v17.0/{FB_PAGE_ID}/feed"
    r = requests.post(url, data={"message":message,"access_token":FB_PAGE_TOKEN}, timeout=20)
    r.raise_for_status()
    return r.json()

def main():
    print("Using HF model:", HF_MODEL)
    for attempt in range(1,6):
        print(f"[Attempt {attempt}] Requesting generation...")
        raw = generate_with_router(PROMPT_AR, HF_MODEL)
        print("Raw preview:", raw[:400].replace("\n"," ") + ("..." if len(raw)>400 else ""))
        title, story, lesson = parse_generated(raw)
        if not (title and story and lesson):
            print("Could not parse generation into title/story/lesson. Retrying...")
            time.sleep(2)
            continue
        bad, w = contains_forbidden(" ".join([title,story,lesson]))
        if bad:
            print("Found forbidden word:", w, " — retrying.")
            time.sleep(2)
            continue
        final = f"{title}\n\n{story}\n\nالعبرة: {lesson}\n\n#قصص\n\nإذا أعجبتك القصة لا تنسى متابعة الصفحة."
        final = tidy_text(final)
        try:
            print("Posting preview (first 300 chars):", final[:300])
            res = post_to_facebook(final)
            print("Posted. Response:", res)
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
    print("Failed after multiple attempts.")
    sys.exit(1)

if __name__ == '__main__':
    main()
