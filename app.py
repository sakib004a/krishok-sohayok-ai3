
import os
import io
import uuid
from datetime import datetime, timezone

import gradio as gr
from PIL import Image
from google import genai
from google.genai import types

from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    DateTime, LargeBinary, ForeignKey, desc
)
from sqlalchemy.orm import declarative_base, sessionmaker


# ==========================================
# 1. CONFIGURATION
# ==========================================

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL", "gemini-3.6-flash"
)

DB_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///krishok_sohayok.db"
)

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY পাওয়া যায়নি। "
        "Render Environment Variables-এ এটি যোগ করুন।"
    )

if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace(
        "postgres://", "postgresql://", 1
    )

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    connect_args=(
        {"check_same_thread": False}
        if DB_URL.startswith("sqlite")
        else {}
    )
)

DB = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

ai = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=60000)
)


# ==========================================
# 2. DATABASE
# ==========================================

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True)
    browser_id = Column(String(80), index=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    image_data = Column(LargeBinary)
    report = Column(Text, nullable=False)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    browser_id = Column(String(80), index=True, nullable=False)
    scope = Column(String(20), index=True, nullable=False)
    analysis_id = Column(
        Integer,
        ForeignKey("analyses.id"),
        nullable=True,
        index=True
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


Base.metadata.create_all(engine)


# ==========================================
# 3. AGRICULTURE PROMPT
# ==========================================

PROMPT = """
তুমি বাংলাদেশের কৃষকদের জন্য একজন অভিজ্ঞ ও দায়িত্বশীল
কৃষি বিশেষজ্ঞ।

তোমার কাজ হলো গাছের ছবি দেখে কৃষককে সরাসরি একটি
পূর্ণাঙ্গ কৃষি রিপোর্ট দেওয়া।

গুরুত্বপূর্ণ নির্দেশনা:

১. শুধু সহজ ও পরিষ্কার বাংলায় উত্তর দেবে।
২. কোনো Checklist, Yes/No, grading, rubric,
   heading verification বা নিজের কাজের মূল্যায়ন লিখবে না।
৩. কোনো ইংরেজি নির্দেশনা বা meta-commentary দেবে না।
৪. ছবিতে যা দেখা যাচ্ছে শুধু সেটিই দৃশ্যমান লক্ষণ হিসেবে বলবে।
৫. রোগ নিশ্চিত না হলে সম্ভাব্য রোগ হিসেবে উল্লেখ করবে।
৬. কোনো তথ্য কল্পনা করবে না।
৭. প্রতিটি শিরোনামের নিচে বাস্তবসম্মত ব্যাখ্যা দেবে।

নিচের ৬টি শিরোনাম ব্যবহার করে রিপোর্ট লিখবে:

### 🌱 গাছের অবস্থা
গাছের সামগ্রিক স্বাস্থ্য ও অবস্থা ব্যাখ্যা করো।

### 🔎 দৃশ্যমান লক্ষণ
পাতা, ফুল, ফল ও কাণ্ডে ছবিতে দেখা লক্ষণগুলো বলো।

### 🦠 সম্ভাব্য সমস্যা
সম্ভাব্য রোগ, পোকামাকড় বা পুষ্টির ঘাটতি ব্যাখ্যা করো।
নিশ্চিত না হলে সম্ভাব্য কারণ হিসেবে উল্লেখ করো।

### 💡 সম্ভাব্য কারণ
সমস্যার সম্ভাব্য কারণগুলো ব্যাখ্যা করো।

### 🧑‍🌾 এখন কী করবেন
কৃষকের জন্য অন্তত ৩টি নিরাপদ ও বাস্তবসম্মত
করণীয় নম্বর দিয়ে লেখো।

### 🛡️ প্রতিরোধ
ভবিষ্যতে সমস্যা প্রতিরোধের অন্তত ২টি উপায় লেখো।

যদি ছবিতে রোগ বোঝা না যায়, তাহলে সেটি স্পষ্টভাবে জানাও।
তারপর নিরাপদ সাধারণ পরিচর্যার পরামর্শ দাও।

রাসায়নিকের নির্দিষ্ট ডোজ অনুমান করবে না।
প্রয়োজনে স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিতে বলবে।

শুধু চূড়ান্ত কৃষি রিপোর্ট লিখবে।
"""


# ==========================================
# 4. USER ID
# ==========================================

def uid(value):
    if isinstance(value, str) and 20 <= len(value) <= 80:
        return value
    return str(uuid.uuid4())


# ==========================================
# 5. GEMINI AI FUNCTION
# ==========================================

def call_ai(contents, max_tokens=1000):

    models_to_try = [MODEL]

    if FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None

    for index, model_name in enumerate(models_to_try):

        parts = []

        try:

            for chunk in ai.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config={
                    "temperature": 0.25,
                    "max_output_tokens": max_tokens
                }
            ):

                chunk_text = getattr(chunk, "text", None)

                if chunk_text:
                    parts.append(chunk_text)
                    yield "".join(parts)

            if parts:
                return

            yield "দুঃখিত, AI থেকে কোনো উত্তর পাওয়া যায়নি।"
            return

        except Exception as exc:

            last_error = exc
            error_text = str(exc).lower()

            temporary_error = (
                "503" in error_text
                or "unavailable" in error_text
                or "high demand" in error_text
            )

            if (
                parts
                or not temporary_error
                or index == len(models_to_try) - 1
            ):
                raise

            yield (
                "⏳ AI সার্ভার ব্যস্ত। "
                "বিকল্প মডেলে চেষ্টা করছি..."
            )

    if last_error:
        raise last_error


# ==========================================
# 6. REPORT VALIDATION
# ==========================================

def report_is_valid(text):

    if not text or len(text.strip()) < 250:
        return False

    lower = text.lower()

    unwanted = [
        "heading 1",
        "heading 2",
        "heading 3",
        "heading 4",
        "heading 5",
        "heading 6",
        "checklist",
        "rubric",
        "at least 2",
        "at least 3",
        "self-evaluation",
        "heading verification"
    ]

    if any(word in lower for word in unwanted):
        return False

    required = [
        "গাছের অবস্থা",
        "দৃশ্যমান লক্ষণ",
        "সম্ভাব্য সমস্যা",
        "সম্ভাব্য কারণ",
        "এখন কী করবেন",
        "প্রতিরোধ"
    ]

    return all(word in text for word in required)


# ==========================================
# 7. IMAGE COMPRESSION
# ==========================================

def image_bytes(img):

    buffer = io.BytesIO()

    img.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=80,
        optimize=True
    )

    return buffer.getvalue()


# ==========================================
# 8. HISTORY
# ==========================================

def history_text(browser_id):

    db = DB()

    try:

        rows = (
            db.query(Analysis)
            .filter_by(browser_id=uid(browser_id))
            .order_by(desc(Analysis.id))
            .limit(30)
            .all()
        )

        if not rows:
            return "এখনো কোনো রিপোর্ট সংরক্ষিত হয়নি।"

        return (
            "### 🗂️ আপনার আগের রিপোর্ট\n"
            + "\n".join(
                f"- **রিপোর্ট #{r.id}** · "
                f"{r.created_at.strftime('%d-%m-%Y %I:%M %p') if r.created_at else ''}  \n"
                f"  {r.report.replace('#', '').replace(chr(10), ' ')[:90]}…"
                for r in rows
            )
        )

    finally:
        db.close()


def choices(browser_id):

    db = DB()

    try:

        rows = (
            db.query(Analysis)
            .filter_by(browser_id=uid(browser_id))
            .order_by(desc(Analysis.id))
            .limit(30)
            .all()
        )

        opts = [
            (
                f"রিপোর্ট #{r.id} · "
                f"{r.created_at.strftime('%d-%m %H:%M') if r.created_at else ''}",
                r.id
            )
            for r in rows
        ]

        return gr.Dropdown(
            choices=opts,
            value=opts[0][1] if opts else None
        )

    finally:
        db.close()


# ==========================================
# 9. PLANT IMAGE ANALYSIS
# ==========================================

def analyze(img, browser_id):

    browser_id = uid(browser_id)

    if img is None:

        yield (
            "⚠️ আগে গাছের ছবি দিন।",
            None,
            history_text(browser_id),
            choices(browser_id)
        )
        return

    try:

        img = img.convert("RGB")
        img.thumbnail((768, 768))

        stored_image = image_bytes(img)

        api_image = Image.open(
            io.BytesIO(stored_image)
        )

        old_history = history_text(browser_id)
        old_choices = choices(browser_id)

        yield (
            "⏳ ছবি বিশ্লেষণ করা হচ্ছে। অনুগ্রহ করে অপেক্ষা করুন...",
            None,
            old_history,
            old_choices
        )

        prompt = (
            PROMPT
            + "\nসংযুক্ত ছবিটি বিশ্লেষণ করে ছয়টি শিরোনামে "
            + "পূর্ণাঙ্গ কৃষি রিপোর্ট দাও। "
            + "কোনো Checklist বা Yes/No উত্তর দেবে না।"
        )

        final = ""

        # First attempt
        for answer in call_ai(
            [prompt, api_image],
            1400
        ):

            final = answer

            yield (
                final,
                None,
                old_history,
                old_choices
            )

        # Retry if report format is invalid
        if not report_is_valid(final):

            retry_prompt = (
                PROMPT
                + "\nআগের উত্তরটি সঠিক কৃষি রিপোর্ট ছিল না। "
                + "এবার সরাসরি ছয়টি বাংলা শিরোনামে "
                + "পূর্ণাঙ্গ কৃষি রিপোর্ট লেখো। "
                + "কোনো Checklist বা Yes/No লিখবে না।"
            )

            final = ""

            for answer in call_ai(
                [retry_prompt, api_image],
                1400
            ):

                final = answer

                yield (
                    final,
                    None,
                    old_history,
                    old_choices
                )

        # Final validation
        if not report_is_valid(final):

            yield (
                "### ⚠️ পূর্ণাঙ্গ রিপোর্ট পাওয়া যায়নি\n\n"
                "AI সঠিক রিপোর্ট দিতে পারেনি। "
                "আরেকটি পরিষ্কার ছবি দিয়ে আবার চেষ্টা করুন।",
                None,
                old_history,
                old_choices
            )

            return

        # Save valid report
        db = DB()

        try:

            rec = Analysis(
                browser_id=browser_id,
                image_data=stored_image,
                report=final
            )

            db.add(rec)
            db.commit()
            db.refresh(rec)

            new_id = rec.id

        finally:
            db.close()

        yield (
            final,
            new_id,
            history_text(browser_id),
            choices(browser_id)
        )

    except Exception as e:

        yield (
            f"### ⚠️ বিশ্লেষণ সম্পন্ন হয়নি\n\n"
            f"`{str(e)[:300]}`",
            None,
            history_text(browser_id),
            choices(browser_id)
        )


# ==========================================
# 10. CHAT HISTORY
# ==========================================

def recent_messages(
    db,
    bid,
    scope,
    aid=None,
    limit=10
):

    q = db.query(ChatMessage).filter_by(
        browser_id=bid,
        scope=scope
    )

    if scope == "analysis":
        q = q.filter_by(analysis_id=aid)

    return list(
        reversed(
            q.order_by(desc(ChatMessage.id))
            .limit(limit)
            .all()
        )
    )


# ==========================================
# 11. AI CHAT REPLY
# ==========================================

def reply(message, bid, scope, aid=None):

    bid = uid(bid)

    if not message or not message.strip():
        return "আপনার প্রশ্নটি লিখুন।"

    db = DB()

    try:

        prompt = [
            PROMPT,
            "বাংলাদেশের কৃষকের উপযোগী সহজ, "
            "সংক্ষিপ্ত ও বাস্তবসম্মত বাংলায় উত্তর দাও।"
        ]

        if scope == "analysis" and aid:

            report = (
                db.query(Analysis)
                .filter_by(
                    id=int(aid),
                    browser_id=bid
                )
                .first()
            )

            if report:
                prompt.append(
                    "এই ছবির রিপোর্টের প্রেক্ষাপট:\n"
                    + report.report[:4000]
                )

        prompt.append("আগের কথোপকথন:")

        for m in recent_messages(
            db, bid, scope, aid
        ):

            prompt.append(
                ("কৃষক: " if m.role == "user" else "সহায়ক: ")
                + m.content
            )

        prompt.append(
            "নতুন প্রশ্ন: " + message.strip()
        )

        answer = ""

        for answer in call_ai(
            "\n\n".join(prompt),
            700
        ):
            pass

        db.add_all([

            ChatMessage(
                browser_id=bid,
                scope=scope,
                analysis_id=aid,
                role="user",
                content=message.strip()
            ),

            ChatMessage(
                browser_id=bid,
                scope=scope,
                analysis_id=aid,
                role="assistant",
                content=answer
            )

        ])

        db.commit()

        return answer

    except Exception as e:

        return (
            f"⚠️ উত্তর তৈরি করা যায়নি: {str(e)[:300]}"
        )

    finally:
        db.close()


# ==========================================
# 12. GENERAL CHAT
# ==========================================

def general_chat(message, bid, chat):

    answer = reply(
        message,
        bid,
        "general"
    )

    chat = list(chat or [])

    chat += [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]

    return chat, ""


# ==========================================
# 13. REPORT CHAT
# ==========================================

def report_chat(message, bid, aid, chat):

    chat = list(chat or [])

    if not aid:

        chat.append({
            "role": "assistant",
            "content": (
                "আগে ছবি বিশ্লেষণ করুন অথবা "
                "History থেকে রিপোর্ট খুলুন।"
            )
        )

        return chat, ""

    answer = reply(
        message,
        bid,
        "analysis",
        aid
    )

    chat += [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]

    return chat, ""


# ==========================================
# 14. LOAD GENERAL CHAT
# ==========================================

def load_general(bid):

    db = DB()

    try:

        return [
            {
                "role": m.role,
                "content": m.content
            }
            for m in recent_messages(
                db,
                uid(bid),
                "general",
                limit=40
            )
        ]

    finally:
        db.close()


# ==========================================
# 15. CLEAR GENERAL CHAT
# ==========================================

def clear_general(bid):

    db = DB()

    try:

        (
            db.query(ChatMessage)
            .filter_by(
                browser_id=uid(bid),
                scope="general"
            )
            .delete()
        )

        db.commit()

    finally:
        db.close()

    return []


# ==========================================
# 16. OPEN OLD REPORT
# ==========================================

def open_report(selected, bid):

    if not selected:
        return "রিপোর্ট নির্বাচন করুন।", None

    db = DB()

    try:

        row = (
            db.query(Analysis)
            .filter_by(
                id=int(selected),
                browser_id=uid(bid)
            )
            .first()
        )

        return (
            (row.report, row.id)
            if row
            else ("রিপোর্ট পাওয়া যায়নি।", None)
        )

    finally:
        db.close()


# ==========================================
# 17. CSS DESIGN
# ==========================================

CSS = r'''
:root{
    color-scheme:light!important
}

.gradio-container{
    max-width:1180px!important;
    margin:auto!important;
    background:#f4f7f1!important;
    color:#18251d!important;
    font-family:'Noto Sans Bengali','Hind Siliguri',sans-serif!important
}

body,.main,.wrap{
    background:#f4f7f1!important;
    color:#18251d!important
}

footer{
    display:none!important
}

#hero{
    background:linear-gradient(
        120deg,#103b2b,#1d6846 68%,#2d8054
    );
    color:white!important;
    border-radius:24px;
    padding:26px 28px;
    margin:8px 0 18px
}

#hero h1,#hero p{
    color:white!important
}

#hero h1{
    font-size:clamp(28px,4vw,40px);
    margin:8px 0
}

#hero p{
    line-height:1.8
}

.panel{
    background:white!important;
    border:1px solid #e1e9df!important;
    border-radius:18px!important;
    padding:16px!important
}

.section-title{
    color:#173e2d!important;
    font-size:20px;
    font-weight:800
}

.section-sub{
    color:#56685c!important;
    font-size:13px
}

#report,#report *{
    background:#fff!important;
    color:#18251d!important
}

#report{
    border:1px solid #dce6dc!important;
    border-radius:18px!important;
    padding:20px!important;
    min-height:230px
}

#report h1,#report h2,#report h3{
    color:#17623e!important
}

#report p,#report li{
    line-height:1.85!important
}

button.primary{
    background:#176b45!important;
    color:#fff!important;
    border:0!important;
    border-radius:12px!important
}

textarea,input{
    background:#fff!important;
    color:#18251d!important
}

label,.prose,.markdown{
    color:#18251d!important
}

#footer-note{
    text-align:center;
    color:#66756a!important;
    font-size:12px;
    padding:18px
}

@media(max-width:700px){
    #hero{
        padding:20px 16px
    }

    .panel{
        padding:12px!important
    }
}
'''


# ==========================================
# 18. GRADIO APPLICATION
# ==========================================

with gr.Blocks(
    title="কৃষক সহায়ক | AI কৃষি সহকারী"
) as demo:

    browser_id = gr.BrowserState(
        default_value=lambda: str(uuid.uuid4()),
        storage_key="krishok_sohayok_anon_id_v2",
        secret=os.getenv(
            "BROWSER_STATE_SECRET",
            "krishok-sohayok-browser-state-v2"
        )
    )

    current_id = gr.State(None)

    gr.HTML("""
    <div id="hero">
        <div style="
            font-size:12px;
            font-weight:800;
            letter-spacing:1.4px;
            color:#d9f2a8
        ">
            SMART FARMING • AI POWERED
        </div>

        <h1>🌾 কৃষক সহায়ক</h1>

        <p>
            গাছের ছবি দিয়ে স্বাস্থ্য পরীক্ষা করুন,
            আগের রিপোর্ট দেখুন,
            অথবা কৃষি বিষয়ে সরাসরি AI-কে প্রশ্ন করুন।
        </p>
    </div>
    """)

    with gr.Tabs():

        # PHOTO ANALYSIS

        with gr.Tab("📸 ছবি দিয়ে পরীক্ষা"):

            with gr.Row():

                with gr.Column(
                    scale=5,
                    elem_classes=["panel"]
                ):

                    gr.HTML(
                        '<div class="section-title">গাছের ছবি দিন</div>'
                        '<p class="section-sub">'
                        'পাতা, ফল, ফুল, কাণ্ড বা পুরো গাছের ছবি '
                        'Upload করুন অথবা Camera ব্যবহার করুন।'
                        '</p>'
                    )

                    plant = gr.Image(
                        sources=["upload", "webcam"],
                        type="pil",
                        label="ছবি আপলোড / ক্যামেরা",
                        height=290
                    )

                    scan = gr.Button(
                        "🔍 গাছের স্বাস্থ্য পরীক্ষা করুন",
                        variant="primary"
                    )

                    gr.Markdown(
                        "**ভালো ফলাফলের জন্য:** "
                        "পরিষ্কার আলোতে আক্রান্ত অংশের কাছ থেকে ছবি তুলুন।"
                    )

                with gr.Column(scale=7):

                    gr.HTML(
                        '<div class="section-title">'
                        '🩺 AI কৃষি ডাক্তারের রিপোর্ট'
                        '</div>'
                        '<p class="section-sub">'
                        'সাদা ব্যাকগ্রাউন্ডে কালো লেখা।'
                        '</p>'
                    )

                    report = gr.Markdown(
                        "আপনার রিপোর্ট এখানে দেখা যাবে।",
                        elem_id="report"
                    )

                    gr.Markdown(
                        "### 💬 এই রিপোর্ট সম্পর্কে প্রশ্ন করুন"
                    )

                    rchat = gr.Chatbot(
                        height=220,
                        label="রিপোর্ট চ্যাট"
                    )

                    with gr.Row():

                        rmsg = gr.Textbox(
                            placeholder="যেমন: এই রোগের জন্য কী করব?",
                            show_label=False,
                            scale=5
                        )

                        rsend = gr.Button(
                            "পাঠান",
                            variant="primary",
                            scale=1
                        )

        # GENERAL AGRICULTURE CHAT

        with gr.Tab("🌿 কৃষি চ্যাট"):

            gr.HTML(
                '<div class="section-title">'
                'আপনার কৃষি বিষয়ক যেকোনো প্রশ্ন করুন'
                '</div>'
                '<p class="section-sub">'
                'ছবি ছাড়াই গাছ, পাতা, ফুল, ফল, কাণ্ড, রোগ, '
                'সার, পানি ও পরিচর্যা নিয়ে প্রশ্ন করুন।'
                '</p>'
            )

            gchat = gr.Chatbot(
                height=420,
                label="AI কৃষি সহকারী"
            )

            with gr.Row():

                gmsg = gr.Textbox(
                    placeholder="যেমন: আমার লেবু গাছের পাতা শুকিয়ে যাচ্ছে, কী করব?",
                    show_label=False,
                    scale=5
                )

                gsend = gr.Button(
                    "পাঠান",
                    variant="primary",
                    scale=1
                )

            gclear = gr.Button("চ্যাট মুছুন")

        # REPORT HISTORY

        with gr.Tab("🗂️ আমার রিপোর্ট"):

            gr.HTML(
                '<div class="section-title">'
                'আপনার আগের Analysis History'
                '</div>'
                '<p class="section-sub">'
                'এই Browser-এ সংরক্ষিত রিপোর্টগুলো এখানে পাবেন।'
                '</p>'
            )

            htext = gr.Markdown()

            hrefresh = gr.Button("↻ History Refresh")

            hselect = gr.Dropdown(
                label="রিপোর্ট নির্বাচন করুন",
                choices=[]
            )

            hopen = gr.Button(
                "রিপোর্ট খুলুন",
                variant="primary"
            )

            hreport = gr.Markdown(
                "নির্বাচিত রিপোর্ট এখানে দেখা যাবে।",
                elem_id="report"
            )

    gr.HTML(
        '<div id="footer-note">'
        'কৃষক সহায়ক • AI-ভিত্তিক প্রাথমিক কৃষি পরামর্শ • '
        'গুরুত্বপূর্ণ ক্ষেত্রে স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।'
        '</div>'
    )


    # ======================================
    # 19. BUTTON EVENTS
    # ======================================

    scan.click(
        analyze,
        [plant, browser_id],
        [report, current_id, htext, hselect],
        show_progress="full"
    )

    rsend.click(
        report_chat,
        [rmsg, browser_id, current_id, rchat],
        [rchat, rmsg],
        show_progress="minimal"
    )

    rmsg.submit(
        report_chat,
        [rmsg, browser_id, current_id, rchat],
        [rchat, rmsg],
        show_progress="minimal"
    )

    gsend.click(
        general_chat,
        [gmsg, browser_id, gchat],
        [gchat, gmsg],
        show_progress="minimal"
    )

    gmsg.submit(
        general_chat,
        [gmsg, browser_id, gchat],
        [gchat, gmsg],
        show_progress="minimal"
    )

    gclear.click(
        clear_general,
        [browser_id],
        [gchat]
    )

    hrefresh.click(
        lambda bid: (
            history_text(bid),
            choices(bid)
        ),
        [browser_id],
        [htext, hselect]
    )

    hopen.click(
        open_report,
        [hselect, browser_id],
        [hreport, current_id]
    )

    demo.load(
        lambda bid: (
            history_text(bid),
            choices(bid),
            load_general(bid)
        ),
        [browser_id],
        [htext, hselect, gchat]
    )


# ==========================================
# 20. START SERVER
# ==========================================

if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=2
    ).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "10000")),
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="lime",
            neutral_hue="slate"
        ),
        css=CSS
    )
