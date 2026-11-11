import os, io, uuid
from datetime import datetime, timezone
import gradio as gr
from PIL import Image
from google import genai
from google.genai import types
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, LargeBinary, ForeignKey, desc
from sqlalchemy.orm import declarative_base, sessionmaker

API_KEY = os.getenv('GEMINI_API_KEY')
MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.8-flash')
FALLBACK_MODEL = os.getenv('GEMINI_FALLBACK_MODEL', 'gemini-3.6-flash')
DB_URL = os.getenv('DATABASE_URL', 'sqlite:///krishok_sohayok.db')
if not API_KEY:
    raise RuntimeError('GEMINI_API_KEY পাওয়া যায়নি। Render Environment Variables-এ এটি যোগ করুন।')
if DB_URL.startswith('postgres://'):
    DB_URL = DB_URL.replace('postgres://', 'postgresql://', 1)
engine = create_engine(DB_URL, pool_pre_ping=True, connect_args={'check_same_thread': False} if DB_URL.startswith('sqlite') else {})
DB = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()
# A bounded request timeout prevents requests from hanging indefinitely.
ai = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=60000))

class Analysis(Base):
    __tablename__ = 'analyses'
    id = Column(Integer, primary_key=True)
    browser_id = Column(String(80), index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    image_data = Column(LargeBinary)
    report = Column(Text, nullable=False)

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(Integer, primary_key=True)
    browser_id = Column(String(80), index=True, nullable=False)
    scope = Column(String(20), index=True, nullable=False)
    analysis_id = Column(Integer, ForeignKey('analyses.id'), nullable=True, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(engine)

PROMPT = '''তুমি বাংলাদেশের কৃষকদের জন্য দায়িত্বশীল কৃষি সহায়ক। সহজ, প্রাঞ্জল বাংলায় উত্তর দাও। ছবির বাইরে থেকে লক্ষণ কল্পনা করবে না। নিশ্চিত না হলে সম্ভাব্য কারণ বলবে, নিশ্চিত রোগ বলবে না। উত্তর সাজাও: ### 🌱 গাছের অবস্থা, ### 🔎 দৃশ্যমান লক্ষণ, ### 🦠 সম্ভাব্য সমস্যা, ### 💡 সম্ভাব্য কারণ, ### 🧑‍🌾 এখন কী করবেন, ### 🛡️ প্রতিরোধ। রাসায়নিকের নির্দিষ্ট ডোজ অনুমান করবে না; পণ্যের লেবেল ও স্থানীয় কৃষি কর্মকর্তার নির্দেশনা মানতে বলবে। এটি প্রাথমিক AI পরামর্শ, নিশ্চিত ল্যাব-পরীক্ষা নয়।'''

def uid(value):
    return value if isinstance(value, str) and 20 <= len(value) <= 80 else str(uuid.uuid4())

def call_ai(contents, max_tokens=900):
    """Stream the answer; if the primary model returns 503 before any text,
    make one attempt with a second supported Flash model.
    The Google GenAI SDK also retries transient HTTP errors automatically.
    """
    models_to_try = [MODEL]
    if FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None
    for model_index, model_name in enumerate(models_to_try):
        parts = []
        try:
            for chunk in ai.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config={'temperature': 0.25, 'max_output_tokens': max_tokens},
            ):
                chunk_text = getattr(chunk, 'text', None)
                if chunk_text:
                    parts.append(chunk_text)
                    yield ''.join(parts)
            if parts:
                return
            yield 'দুঃখিত, AI থেকে কোনো উত্তর পাওয়া যায়নি। আবার চেষ্টা করুন।'
            return
        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()
            is_temporary_capacity_error = (
                '503' in error_text or 'unavailable' in error_text or
                'high demand' in error_text
            )
            # Do not restart after partial output; that could duplicate text.
            if parts or not is_temporary_capacity_error or model_index == len(models_to_try) - 1:
                raise
            yield f'⏳ মূল AI মডেল ব্যস্ত। বিকল্প মডেলে চেষ্টা করছি…'

    if last_error:
        raise last_error

def image_bytes(img):
    b=io.BytesIO(); img.convert('RGB').save(b, format='JPEG', quality=80, optimize=True); return b.getvalue()

def history_text(browser_id):
    db=DB()
    try:
        rows=db.query(Analysis).filter_by(browser_id=uid(browser_id)).order_by(desc(Analysis.id)).limit(30).all()
        if not rows: return 'এখনো কোনো রিপোর্ট সংরক্ষিত হয়নি।'
        return '### 🗂️ আপনার আগের রিপোর্ট\n'+'\n'.join(
            f"- **রিপোর্ট #{r.id}** · {r.created_at.strftime('%d-%m-%Y %I:%M %p') if r.created_at else ''}  \n  {r.report.replace('#','').replace(chr(10),' ')[:90]}…" for r in rows)
    finally: db.close()

def choices(browser_id):
    db=DB()
    try:
        rows=db.query(Analysis).filter_by(browser_id=uid(browser_id)).order_by(desc(Analysis.id)).limit(30).all()
        opts=[(f"রিপোর্ট #{r.id} · {r.created_at.strftime('%d-%m %H:%M') if r.created_at else ''}",r.id) for r in rows]
        return gr.Dropdown(choices=opts, value=opts[0][1] if opts else None)
    finally: db.close()

def analyze(img, browser_id):
    browser_id=uid(browser_id)
    if img is None:
        yield '⚠️ আগে গাছ, পাতা, ফুল বা ফলের ছবি দিন।', None, history_text(browser_id), choices(browser_id)
        return
    try:
        img=img.convert('RGB')
        img.thumbnail((768,768))
        # Compress before sending to Gemini to reduce upload time and payload size.
        stored_image=image_bytes(img)
        api_image=Image.open(io.BytesIO(stored_image))
        old_history=history_text(browser_id)
        old_choices=choices(browser_id)
        yield '⏳ ছবি বিশ্লেষণ করা হচ্ছে… অনুগ্রহ করে অপেক্ষা করুন।', None, old_history, old_choices
        final=''
        for final in call_ai([PROMPT+'\nছবিটি বিশ্লেষণ করে সংক্ষিপ্ত ও ব্যবহারিক পরামর্শ দাও।',api_image],750):
            yield final, None, old_history, old_choices
        db=DB()
        try:
            rec=Analysis(browser_id=browser_id,image_data=stored_image,report=final)
            db.add(rec); db.commit(); db.refresh(rec); new_id=rec.id
        finally: db.close()
        yield final,new_id,history_text(browser_id),choices(browser_id)
    except Exception as e:
        yield f'### ⚠️ বিশ্লেষণ সম্পন্ন হয়নি\nGemini সার্ভার ব্যস্ত, নেটওয়ার্ক ধীর, অথবা API configuration-এ সমস্যা হতে পারে। কিছুক্ষণ পর আবার চেষ্টা করুন।\n\n`{str(e)[:300]}`',None,history_text(browser_id),choices(browser_id)

def recent_messages(db, bid, scope, aid=None, limit=10):
    q=db.query(ChatMessage).filter_by(browser_id=bid,scope=scope)
    if scope=='analysis': q=q.filter_by(analysis_id=aid)
    return list(reversed(q.order_by(desc(ChatMessage.id)).limit(limit).all()))

def reply(message,bid,scope,aid=None):
    bid=uid(bid)
    if not message or not message.strip(): return 'আপনার প্রশ্নটি লিখুন।'
    db=DB()
    try:
        prompt=[PROMPT,'বাংলাদেশের কৃষকের উপযোগী সংক্ষিপ্ত, বাস্তবসম্মত বাংলায় উত্তর দাও।']
        if scope=='analysis' and aid:
            report=db.query(Analysis).filter_by(id=int(aid),browser_id=bid).first()
            if report: prompt.append('এই ছবির রিপোর্টের প্রেক্ষাপট:\n'+report.report[:4000])
        prompt.append('আগের কথোপকথন:')
        for m in recent_messages(db,bid,scope,aid): prompt.append(('কৃষক: ' if m.role=='user' else 'সহায়ক: ')+m.content)
        prompt.append('নতুন প্রশ্ন: '+message.strip())
        answer=''
        for answer in call_ai('\n\n'.join(prompt),700): pass
        db.add_all([ChatMessage(browser_id=bid,scope=scope,analysis_id=aid,role='user',content=message.strip()),
                    ChatMessage(browser_id=bid,scope=scope,analysis_id=aid,role='assistant',content=answer)])
        db.commit(); return answer
    except Exception as e: return f'⚠️ উত্তর তৈরি করা যায়নি: {str(e)[:300]}'
    finally: db.close()

def general_chat(message,bid,chat):
    answer=reply(message,bid,'general'); chat=list(chat or [])
    chat += [{'role':'user','content':message},{'role':'assistant','content':answer}]
    return chat,''

def report_chat(message,bid,aid,chat):
    chat=list(chat or [])
    if not aid:
        chat.append({'role':'assistant','content':'আগে ছবি বিশ্লেষণ করুন অথবা History থেকে রিপোর্ট খুলুন।'})
        return chat,''
    answer=reply(message,bid,'analysis',aid)
    chat += [{'role':'user','content':message},{'role':'assistant','content':answer}]
    return chat,''

def load_general(bid):
    db=DB()
    try: return [{'role':m.role,'content':m.content} for m in recent_messages(db,uid(bid),'general',limit=40)]
    finally: db.close()

def clear_general(bid):
    db=DB()
    try: db.query(ChatMessage).filter_by(browser_id=uid(bid),scope='general').delete(); db.commit()
    finally: db.close()
    return []

def open_report(selected,bid):
    if not selected: return 'রিপোর্ট নির্বাচন করুন।',None
    db=DB()
    try:
        row=db.query(Analysis).filter_by(id=int(selected),browser_id=uid(bid)).first()
        return (row.report,row.id) if row else ('রিপোর্ট পাওয়া যায়নি।',None)
    finally: db.close()

CSS=r'''
:root{color-scheme:light!important}.gradio-container{max-width:1180px!important;margin:auto!important;background:#f4f7f1!important;color:#18251d!important;font-family:'Noto Sans Bengali','Hind Siliguri',sans-serif!important}body,.main,.wrap{background:#f4f7f1!important;color:#18251d!important}footer{display:none!important}#hero{background:linear-gradient(120deg,#103b2b,#1d6846 68%,#2d8054);color:white!important;border-radius:24px;padding:26px 28px;margin:8px 0 18px}#hero h1,#hero p{color:white!important}#hero h1{font-size:clamp(28px,4vw,40px);margin:8px 0}#hero p{line-height:1.8}.panel{background:white!important;border:1px solid #e1e9df!important;border-radius:18px!important;padding:16px!important}.section-title{color:#173e2d!important;font-size:20px;font-weight:800}.section-sub{color:#56685c!important;font-size:13px}#report,#report *{background:#fff!important;color:#18251d!important}#report{border:1px solid #dce6dc!important;border-radius:18px!important;padding:20px!important;min-height:230px}#report h1,#report h2,#report h3{color:#17623e!important}#report p,#report li{line-height:1.85!important}button.primary{background:#176b45!important;color:#fff!important;border:0!important;border-radius:12px!important}textarea,input{background:#fff!important;color:#18251d!important}label,.prose,.markdown{color:#18251d!important}#footer-note{text-align:center;color:#66756a!important;font-size:12px;padding:18px}@media(max-width:700px){#hero{padding:20px 16px}.panel{padding:12px!important}}
'''

with gr.Blocks(title='কৃষক সহায়ক | AI কৃষি সহকারী') as demo:
    browser_id=gr.BrowserState(default_value=lambda:str(uuid.uuid4()),storage_key='krishok_sohayok_anon_id_v2',secret=os.getenv('BROWSER_STATE_SECRET','krishok-sohayok-browser-state-v2'))
    current_id=gr.State(None)
    gr.HTML('''<div id="hero"><div style="font-size:12px;font-weight:800;letter-spacing:1.4px;color:#d9f2a8">SMART FARMING • AI POWERED</div><h1>🌾 কৃষক সহায়ক</h1><p>গাছের ছবি দিয়ে স্বাস্থ্য পরীক্ষা করুন, আগের রিপোর্ট দেখুন, অথবা কৃষি বিষয়ে সরাসরি AI-কে প্রশ্ন করুন।</p></div>''')
    with gr.Tabs():
        with gr.Tab('📸 ছবি দিয়ে পরীক্ষা'):
            with gr.Row():
                with gr.Column(scale=5,elem_classes=['panel']):
                    gr.HTML('<div class="section-title">গাছের ছবি দিন</div><p class="section-sub">পাতা, ফল, ফুল, কাণ্ড বা পুরো গাছের ছবি Upload করুন অথবা Camera ব্যবহার করুন।</p>')
                    plant=gr.Image(sources=['upload','webcam'],type='pil',label='ছবি আপলোড / ক্যামেরা',height=290)
                    scan=gr.Button('🔍 গাছের স্বাস্থ্য পরীক্ষা করুন',variant='primary')
                    gr.Markdown('**ভালো ফলাফলের জন্য:** পরিষ্কার আলোতে আক্রান্ত অংশের কাছ থেকে ছবি তুলুন।')
                with gr.Column(scale=7):
                    gr.HTML('<div class="section-title">🩺 AI কৃষি ডাক্তারের রিপোর্ট</div><p class="section-sub">সাদা ব্যাকগ্রাউন্ডে কালো লেখা।</p>')
                    report=gr.Markdown('আপনার রিপোর্ট এখানে দেখা যাবে।',elem_id='report')
                    gr.Markdown('### 💬 এই রিপোর্ট সম্পর্কে প্রশ্ন করুন')
                    rchat=gr.Chatbot(height=220,label='রিপোর্ট চ্যাট')
                    with gr.Row():
                        rmsg=gr.Textbox(placeholder='যেমন: এই রোগের জন্য কী করব?',show_label=False,scale=5)
                        rsend=gr.Button('পাঠান',variant='primary',scale=1)
        with gr.Tab('🌿 কৃষি চ্যাট'):
            gr.HTML('<div class="section-title">আপনার কৃষি বিষয়ক যেকোনো প্রশ্ন করুন</div><p class="section-sub">ছবি ছাড়াই গাছ, পাতা, ফুল, ফল, কাণ্ড, রোগ, সার, পানি ও পরিচর্যা নিয়ে প্রশ্ন করুন।</p>')
            gchat=gr.Chatbot(height=420,label='AI কৃষি সহকারী')
            with gr.Row():
                gmsg=gr.Textbox(placeholder='যেমন: আমার লেবু গাছের পাতা শুকিয়ে যাচ্ছে, কী করব?',show_label=False,scale=5)
                gsend=gr.Button('পাঠান',variant='primary',scale=1)
            gclear=gr.Button('চ্যাট মুছুন')
        with gr.Tab('🗂️ আমার রিপোর্ট'):
            gr.HTML('<div class="section-title">আপনার আগের Analysis History</div><p class="section-sub">এই Browser-এ সংরক্ষিত রিপোর্টগুলো এখানে পাবেন।</p>')
            htext=gr.Markdown()
            hrefresh=gr.Button('↻ History Refresh')
            hselect=gr.Dropdown(label='রিপোর্ট নির্বাচন করুন',choices=[])
            hopen=gr.Button('রিপোর্ট খুলুন',variant='primary')
            hreport=gr.Markdown('নির্বাচিত রিপোর্ট এখানে দেখা যাবে।',elem_id='report')
    gr.HTML('<div id="footer-note">কৃষক সহায়ক • AI-ভিত্তিক প্রাথমিক কৃষি পরামর্শ • গুরুত্বপূর্ণ ক্ষেত্রে স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।</div>')

    scan.click(analyze,[plant,browser_id],[report,current_id,htext,hselect],show_progress='full')
    rsend.click(report_chat,[rmsg,browser_id,current_id,rchat],[rchat,rmsg],show_progress='minimal')
    rmsg.submit(report_chat,[rmsg,browser_id,current_id,rchat],[rchat,rmsg],show_progress='minimal')
    gsend.click(general_chat,[gmsg,browser_id,gchat],[gchat,gmsg],show_progress='minimal')
    gmsg.submit(general_chat,[gmsg,browser_id,gchat],[gchat,gmsg],show_progress='minimal')
    gclear.click(clear_general,[browser_id],[gchat])
    hrefresh.click(lambda bid:(history_text(bid),choices(bid)),[browser_id],[htext,hselect])
    hopen.click(open_report,[hselect,browser_id],[hreport,current_id])
    demo.load(lambda bid:(history_text(bid),choices(bid),load_general(bid)),[browser_id],[htext,hselect,gchat])

if __name__=='__main__':
    demo.queue(default_concurrency_limit=2).launch(server_name='0.0.0.0',server_port=int(os.getenv('PORT','10000')),theme=gr.themes.Soft(primary_hue='green',secondary_hue='lime',neutral_hue='slate'),css=CSS)
