"""
Career Platform — main Flask application
Three AI tools powered by Lenny's Newsletter archive via RAG.

Tools:
  1. /coach        — AI Career Coach
  2. /interviewer  — AI Mock Interviewer
  3. /scorecard    — Employer Readiness Scorecard

API routes (require X-API-Key header):
  POST /api/coach
  POST /api/interview/start
  POST /api/interview/answer
  GET  /api/interview/report
  POST /api/scorecard

Public routes:
  GET /health
"""

import os
import json
import functools
import anthropic
from flask import Flask, request, jsonify, render_template, session
from rag.retriever import retrieve, build_context, get_chunk_count
from config import ANTHROPIC_API_KEY, SECRET_KEY, RAG_API_KEY

app = Flask(__name__)
app.secret_key = SECRET_KEY
ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def ask_claude(system, user, max_tokens=1200):
    """Single Claude call — returns text string."""
    msg = ai.messages.create(
        model="claude-opus-4-5",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return msg.content[0].text


def parse_json(text):
    """Extract JSON from Claude response, stripping markdown fences."""
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                text = part
                break
    try:
        return json.loads(text)
    except Exception:
        return None


def require_key(f):
    """Decorator: validates X-API-Key header on protected endpoints."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if not key or key != RAG_API_KEY:
            return jsonify({"error": "Unauthorized — invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return wrapper


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/coach")
def coach_page():
    return render_template("coach.html", api_key=RAG_API_KEY)


@app.route("/interviewer")
def interviewer_page():
    session.clear()
    return render_template("interviewer.html", api_key=RAG_API_KEY)


@app.route("/scorecard")
def scorecard_page():
    return render_template("scorecard.html", api_key=RAG_API_KEY)


@app.route("/health")
def health():
    count = get_chunk_count()
    return jsonify({
        "status":  "ok",
        "chunks":  count,
        "tools":   ["career_coach", "mock_interviewer", "scorecard"],
        "version": "1.0.0"
    })


COACH_SYSTEM = """You are a warm, direct career coach helping young people
aged 16-30 in the UK — school leavers, university students, recent graduates,
and career changers. Many come from disadvantaged backgrounds without family
networks or professional connections.

You draw on expert career, hiring, and professional development research.
Your job is to:
- Translate any product-management or startup jargon into plain language
- Give advice specific to their exact situation — age, qualifications, interests, goal
- Acknowledge real barriers (lack of connections, no experience, qualifications gaps) honestly
- Never give generic platitudes — every sentence must be useful
- Be encouraging without being dishonest

Always end your response with exactly 3 specific numbered actions
the person can take THIS WEEK — not "research roles" but
"Search LinkedIn for [specific job title] in [city] and message 2 people
with the word 'junior' in their title asking for a 15-minute call"."""


@app.route("/api/coach", methods=["POST"])
@require_key
def coach_api():
    d = request.json or {}
    age       = d.get("age", "")
    quals     = d.get("quals", "")
    interests = d.get("interests", "")
    goal      = d.get("goal", "")

    situation = (
        f"Age: {age}. "
        f"Highest qualification: {quals}. "
        f"Interests: {interests}. "
        f"Career goal: {goal}."
    )

    query  = f"career advice {age} year old interested in {interests} wants to {goal}"
    chunks = retrieve(query, n=6)
    ctx    = build_context(chunks)

    prompt = (
        f"Person I am helping: {situation}\n\n"
        f"Expert research context:\n{ctx}\n\n"
        f"Give this person personalised, specific career advice. "
        f"Tailor every sentence to their exact situation."
    )

    advice = ask_claude(COACH_SYSTEM, prompt)

    return jsonify({
        "advice":  advice,
        "sources": [{"title": c["title"], "url": c["url"]} for c in chunks[:3]]
    })


@app.route("/api/interview/start", methods=["POST"])
@require_key
def interview_start():
    d    = request.json or {}
    role = d.get("role", "general")

    chunks  = retrieve(
        f"behavioral interview questions {role} what interviewers look for hiring",
        n=5,
        topics=["interviews", "hiring", "careers"]
    )
    ctx = build_context(chunks, max_chars=2000)

    qs_text = ask_claude(
        "Generate exactly 5 behavioural interview questions for the given role. "
        "Return a JSON array of 5 question strings only. No other text, no markdown.",
        f"Role type: {role}\n\nContext on what interviewers value:\n{ctx}"
    )

    questions = parse_json(qs_text)
    if not questions or not isinstance(questions, list) or len(questions) < 3:
        questions = [
            "Tell me about a time you had to work closely with someone who had a very different working style. How did you handle it?",
            "Describe a situation where you had to learn a new skill or piece of knowledge quickly. What did you do?",
            "Tell me about a goal you set for yourself and the steps you took to achieve it.",
            "Describe a time you received critical feedback. How did you respond, and what did you do differently afterwards?",
            "Tell me about a challenge you faced and what you did to overcome it. What was the outcome?"
        ]

    session["qs"]      = questions
    session["scores"]  = []
    session["current"] = 0
    session["role"]    = role

    return jsonify({
        "question": questions[0],
        "num":      1,
        "total":    len(questions)
    })


@app.route("/api/interview/answer", methods=["POST"])
@require_key
def interview_answer():
    d       = request.json or {}
    answer  = d.get("answer", "")
    current = session.get("current", 0)
    qs      = session.get("qs", [])

    if current >= len(qs):
        return jsonify({"error": "Interview already complete — get your report"}), 400

    question = qs[current]

    score_prompt = f"""Question asked: {question}

Candidate's answer: {answer}

Score this answer using the STAR method. Return valid JSON only — no markdown, no explanation.

{{
  "situation":     <1-10>,
  "task":          <1-10>,
  "action":        <1-10>,
  "result":        <1-10>,
  "communication": <1-10>,
  "strength":      "<quote their exact words that worked and explain why>",
  "improvement":   "<give the exact better phrasing they should use next time>"
}}"""

    score_text = ask_claude(
        "You are a professional interview coach using the STAR method. "
        "Score interview answers honestly. Return valid JSON only.",
        score_prompt,
        max_tokens=512
    )

    score = parse_json(score_text)
    if not score:
        score = {
            "situation": 5, "task": 5, "action": 5, "result": 5, "communication": 5,
            "strength":    "You engaged with the question directly.",
            "improvement": "Try to quantify your result — e.g. 'which led to X outcome'."
        }

    scores = session.get("scores", [])
    scores.append({"q": question, "a": answer, "score": score})
    session["scores"]  = scores
    session["current"] = current + 1

    is_final      = (current + 1) >= len(qs)
    next_question = qs[current + 1] if not is_final else None
    next_num      = current + 2      if not is_final else None

    return jsonify({
        "score":         score,
        "is_final":      is_final,
        "next_question": next_question,
        "next_num":      next_num
    })


@app.route("/api/interview/report")
@require_key
def interview_report():
    scores = session.get("scores", [])
    if not scores:
        return jsonify({"error": "No interview data found — start a new session"}), 400

    dim_totals = {
        "situation": 0, "task": 0,
        "action": 0,    "result": 0,
        "communication": 0
    }
    total = 0
    for s in scores:
        sc = s["score"]
        for d in dim_totals:
            val = sc.get(d, 0)
            dim_totals[d] += val
            total += val

    maximum = len(scores) * 50
    weakest = min(dim_totals, key=dim_totals.get)

    return jsonify({
        "total":    total,
        "maximum":  maximum,
        "percent":  round(total / maximum * 100) if maximum else 0,
        "weakest":  weakest,
        "scores":   scores,
        "role":     session.get("role", "")
    })


@app.route("/api/scorecard", methods=["POST"])
@require_key
def scorecard_api():
    d = request.json or {}

    sector = d.get("sector", "")
    level  = d.get("level", "")

    query  = f"job market skills demand employers want {sector} {level} entry level"
    chunks = retrieve(query, n=6, topics=["job-market", "skills", "careers", "general"])
    ctx    = build_context(chunks, max_chars=3000)

    prompt = f"""Student profile:
{json.dumps(d, indent=2)}

Current job market research:
{ctx}

Analyse this student's employment readiness and return valid JSON only.
No markdown fences, no explanation outside the JSON.

{{
  "readiness_score": <integer 0-100>,
  "score_label":     "<Ready to apply|Nearly there|Building foundations|Early stage>",
  "strengths": [
    {{"skill": "<skill name>", "evidence": "<why this is a genuine strength based on their profile>"}}
  ],
  "gaps": [
    {{
      "skill":    "<specific skill name>",
      "severity": "<critical|important|nice-to-have>",
      "action":   "<one specific free action they can start this week>"
    }}
  ],
  "week_1": "<specific focus and action for week 1>",
  "week_2": "<specific focus and action for week 2>",
  "week_3": "<specific focus and action for week 3>",
  "week_4": "<specific measurable action for week 4>",
  "summary":     "<honest 20-word max summary of their current position>",
  "next_target": "<specific type of role, programme or opportunity to apply to first>"
}}"""

    result_text = ask_claude(
        "You are an employment readiness analyst. Be specific and honest. "
        "Return valid JSON only — no markdown, no preamble.",
        prompt,
        max_tokens=1400
    )

    result = parse_json(result_text)
    if not result:
        return jsonify({"error": "Could not parse result", "raw": result_text[:500]}), 500

    return jsonify(result)


_seed_state = {"status": "idle", "log": []}


def _run_seed():
    import pipeline.scraper as _s
    from pipeline.manifest import load, save, mark_done, get_new_urls, summary
    from pipeline.scraper  import scrape_new
    from pipeline.embedder import embed_and_store, get_total_chunks
    from pipeline.all_urls import ALL_ARTICLE_URLS, ALL_PODCAST_URLS
    _s.REQUEST_DELAY = 1.0

    def log(msg):
        print(msg, flush=True)
        _seed_state["log"].append(msg)

    try:
        _seed_state["status"] = "running"
        manifest     = load()
        new_articles = get_new_urls(manifest, ALL_ARTICLE_URLS)
        new_podcasts = [(u, t) for u, t in ALL_PODCAST_URLS
                       if get_new_urls(manifest, [u])]

        if not new_articles and not new_podcasts:
            _seed_state["status"] = "done"
            _seed_state["result"] = {"status": "already_seeded", "chunks": get_total_chunks()}
            return

        log(f"Scraping {len(new_articles)} articles + {len(new_podcasts)} podcasts...")
        chunks = scrape_new(new_articles, new_podcasts)
        log(f"Got {len(chunks)} chunks. Embedding into Supabase...")

        stored = embed_and_store(chunks)

        url_to_chunks = {}
        for c in chunks:
            url_to_chunks.setdefault(c["url"], 0)
            url_to_chunks[c["url"]] += 1

        for url in new_articles:
            count = url_to_chunks.get(url, 0)
            mark_done(manifest, url, count, status="ok" if count else "paywalled")
        for url, _ in new_podcasts:
            count = url_to_chunks.get(url, 0)
            mark_done(manifest, url, count, status="ok" if count else "paywalled")
        save(manifest)

        _seed_state["status"] = "done"
        _seed_state["result"] = {
            "status":  "done",
            "stored":  stored,
            "total":   get_total_chunks(),
            "summary": summary(manifest)
        }
    except Exception as e:
        _seed_state["status"] = "error"
        _seed_state["result"] = {"error": str(e)}


STATIC_CONTENT = [
    {
        "title": "How to pass any first-round job interview",
        "url": "https://www.lennysnewsletter.com/p/how-to-pass-any-first-round-interview",
        "source_type": "article", "topics": "interviews,hiring,careers",
        "content": "The STAR method (Situation, Task, Action, Result) is the gold standard for behavioural interview answers. Interviewers are trained to listen for it. Without structure, even great answers get marked down. Spend 20% of your time on Situation and Task combined — briefly set the scene. Spend 60% on Action — this is what the interviewer is actually evaluating: what YOU did, not what 'we' did as a team. Spend 20% on Result — quantify whenever possible: 'reduced complaints by 40%', 'increased sales by £2k'. Never end without a result."
    },
    {
        "title": "How to pass any first-round job interview",
        "url": "https://www.lennysnewsletter.com/p/how-to-pass-any-first-round-interview",
        "source_type": "article", "topics": "interviews,hiring,careers",
        "content": "Most candidates fail interviews not because they lack skills but because they lack preparation. Do these three things before any interview: (1) Research the company's last 3 announcements or news stories — interviewers notice when you know their world. (2) Prepare 5 STAR stories that cover different skills — team conflict, failure, leadership, achievement under pressure, learning a new skill. (3) Prepare 3 thoughtful questions to ask them — not about salary, about the team's biggest challenge right now."
    },
    {
        "title": "How to use AI in your next job interview",
        "url": "https://www.lennysnewsletter.com/p/how-to-use-ai-in-your-next-job-interview",
        "source_type": "article", "topics": "interviews,hiring,careers,skills",
        "content": "AI tools can dramatically improve interview preparation. Use Claude or ChatGPT to: (1) Generate likely interview questions for a specific job description — paste the JD and ask for the 10 most likely behavioural questions. (2) Practise out loud and record yourself — the camera-off video recording trick helps you spot filler words and poor posture. (3) Get feedback on your STAR answers — paste your answer and ask the AI to score it on clarity, specificity, and impact. (4) Research the company's product strategy using AI to synthesise recent news."
    },
    {
        "title": "State of the product job market",
        "url": "https://www.lennysnewsletter.com/p/state-of-the-product-job-market-in-ee9",
        "source_type": "article", "topics": "job-market,careers,skills",
        "content": "The job market for entry-level and junior roles is more competitive than at any point in the last decade. Employers report receiving 200-400 applications per role. The candidates who break through share three traits: (1) They apply with a tailored cover letter that references something specific about the company. (2) They have a portfolio or side project that demonstrates the skill being hired for. (3) They have one warm introduction — from a current employee, a recruiter, or an event connection. Cold applications alone rarely succeed at volume."
    },
    {
        "title": "State of the product job market",
        "url": "https://www.lennysnewsletter.com/p/state-of-the-product-job-market-part",
        "source_type": "article", "topics": "job-market,careers,skills",
        "content": "Skills employers are currently hiring for in technology roles: data literacy (SQL, Excel, basic analytics), AI prompt engineering and tool use, communication in writing (Slack, docs, async), project coordination, and customer empathy. Hard technical skills like coding are nice-to-have for non-engineering roles but rarely required. What IS required at every level: the ability to work through ambiguity, communicate clearly, and show initiative without being asked. These are harder to fake and harder to teach."
    },
    {
        "title": "What interviewers actually look for when hiring junior candidates",
        "url": "https://www.lennysnewsletter.com/p/how-to-pass-any-first-round-interview",
        "source_type": "article", "topics": "interviews,hiring,careers",
        "content": "Senior hiring managers consistently say the same things when asked what makes a junior candidate stand out: curiosity (they ask good questions and clearly learned something about us), coachability (they respond to pushback by thinking rather than defending), and energy (they seem to genuinely want this, not just any job). Technical skills matter less than you think at the entry level — most companies expect to train you. What they cannot train is attitude. Show you want to learn."
    },
    {
        "title": "How to find your first job with no experience",
        "url": "https://www.lennysnewsletter.com/p/finding-product-market-fit",
        "source_type": "article", "topics": "careers,job-market",
        "content": "No experience is a mindset problem more than a market problem. You have more relevant experience than you think — you just haven't framed it yet. Retail and hospitality jobs teach: customer handling under pressure, team coordination, speed, conflict resolution, and working to targets. Volunteering teaches: project management, stakeholder communication, and resourcefulness. University projects teach: research, deadlines, and collaboration. The job of a CV and cover letter is not to list what you did — it is to translate what you did into the language employers care about."
    },
    {
        "title": "How to network when you know nobody",
        "url": "https://www.lennysnewsletter.com/p/building-community",
        "source_type": "article", "topics": "careers,job-search",
        "content": "Networking works. It feels uncomfortable, but it works. Here is a system that doesn't require confidence: (1) Find 5 people on LinkedIn who have the job you want. Look at their profile and find one thing that's genuinely interesting — a project, a post, a career move. (2) Send a 3-sentence message: what you noticed, one genuine question about their experience, and a soft ask ('would you be open to a 15-minute call?'). (3) Do not ask for a job. Ask for insight. People help when they feel consulted, not recruited. A 20% reply rate is normal and good."
    },
    {
        "title": "How AI is reshaping entry-level jobs",
        "url": "https://www.lennysnewsletter.com/p/how-ai-is-reshaping-the-product-role",
        "source_type": "podcast", "topics": "job-market,skills,careers",
        "content": "AI is not eliminating entry-level jobs — it is changing what entry-level jobs require. The roles disappearing are purely repetitive: data entry, basic report generation, templated content. The roles growing are: AI-assisted analysis, prompt engineering, customer success with AI tools, and roles that require human judgment on top of AI output. The candidates winning right now are those who treat AI as a multiplier: they get 10x more done than their peers using the same tools. Learning to use AI well is the most valuable skill a junior candidate can develop in 2024-2025."
    },
    {
        "title": "How to write a CV that gets past ATS and to a human",
        "url": "https://www.lennysnewsletter.com/p/product-sense",
        "source_type": "article", "topics": "careers,hiring,job-search",
        "content": "Most CVs are rejected before a human reads them. Applicant Tracking Systems (ATS) scan for keyword matches with the job description. To pass: (1) Mirror the exact language of the job description in your CV — if they say 'stakeholder management' use that phrase, not 'working with people'. (2) Keep formatting simple — no tables, no columns, no graphics, no headers in text boxes. (3) Use reverse-chronological order. (4) Quantify everything possible: not 'improved customer satisfaction' but 'improved customer satisfaction score from 3.2 to 4.1 over 6 months'. (5) Keep it to one page if under 5 years experience."
    },
    {
        "title": "The 30-day job search sprint",
        "url": "https://www.lennysnewsletter.com/p/land-your-dream-job-phyl-terry",
        "source_type": "podcast", "topics": "job-search,careers",
        "content": "A structured 30-day sprint beats months of passive job searching. Week 1: Define your target — specific roles, specific companies, specific locations. Do not apply to everything. Week 2: Build your materials — tailored CV, LinkedIn profile optimised with keywords, 3 cover letter templates. Week 3: Apply to 5 roles per day AND reach out to 3 people in your network per day. Do both. Week 4: Follow up, prepare for interviews, and keep the pipeline moving. Most people stop after week 1. The ones who get jobs are the ones who keep going."
    },
    {
        "title": "How to prepare for a technical or skills-based interview",
        "url": "https://www.lennysnewsletter.com/p/how-to-use-ai-in-your-next-job-interview",
        "source_type": "article", "topics": "interviews,skills,careers",
        "content": "For roles requiring demonstrated skills (data analysis, design, writing, coding), many employers now use take-home tasks or live skill tests. How to prepare: (1) Find examples of similar tasks on Glassdoor or by asking your network. (2) Time yourself — employers care about quality under time pressure, not perfection given unlimited time. (3) Show your thinking, not just your answer — a written explanation of your approach impresses more than a polished output with no rationale. (4) Use AI tools in the same way you would on the job — if they'd be available at work, they're fair to use in the test."
    },
    {
        "title": "What good retention looks like and what employers want",
        "url": "https://www.lennysnewsletter.com/p/what-is-good-retention-issue-29",
        "source_type": "article", "topics": "job-market,careers,general",
        "content": "Employers hire people they think will stay and grow. To signal you're a long-term candidate: (1) In interviews, show genuine interest in the company's mission — not just the role. (2) Ask about growth: 'What does success look like in this role after 12 months?' shows you're thinking about trajectory. (3) Reference the company's specific products or customers — it shows you've done homework. (4) If asked 'where do you see yourself in 5 years?' — answer honestly but frame it as growth within a company like this one. Employers want ambition, but not so much ambition that you'll leave in 6 months."
    },
    {
        "title": "How to handle rejection and keep going",
        "url": "https://www.lennysnewsletter.com/p/how-to-increase-your-retention-issue",
        "source_type": "article", "topics": "careers,job-search",
        "content": "Most job seekers give up after 5-10 rejections. The data shows that successful first-time job seekers receive on average 15-25 rejections before their first offer. Rejection is a numbers and timing problem, not a quality problem. Things that help: (1) Ask for feedback — most companies won't give it, but 20% will and it's gold. (2) Keep a log of applications so you can spot patterns. (3) Improve one thing each week — your opening sentence, your answer to 'tell me about yourself', your LinkedIn headline. (4) Celebrate effort, not outcomes — sending 5 quality applications is a win regardless of what comes back."
    },
    {
        "title": "How to use behavioural science to get hired faster",
        "url": "https://www.lennysnewsletter.com/p/how-to-use-behavioral-science-to",
        "source_type": "article", "topics": "interviews,hiring,careers",
        "content": "Interviewers make hiring decisions based on feeling as much as fact. Three principles from behavioural science that help: (1) The peak-end rule — people remember the peak moment and the ending of an experience. Make sure your best story is memorable and end every interview strongly. (2) Social proof — interviewers are reassured by credibility signals. Name-drop relevant projects, courses, or experiences naturally. (3) Commitment and consistency — if an interviewer has praised something you said, gently reference it later. It anchors their positive impression. These are not tricks — they are how human memory and judgment work."
    },
    {
        "title": "How to negotiate your salary as a junior candidate",
        "url": "https://www.lennysnewsletter.com/p/negotiating-your-salary",
        "source_type": "article", "topics": "careers,compensation,negotiation",
        "content": "Most candidates leave money on the table because they're afraid to ask. Three rules for junior negotiation: (1) Never give a number first — when asked your expected salary, respond with 'I'd like to understand the full package first — what range have you budgeted for this role?'. (2) Always counter the first offer, even if it's good — 'Thank you, I'm excited. Is there flexibility on the base?' adds 5-15% in most cases without risk. (3) Negotiate the whole package — signing bonus, start date, holiday days, learning budget, and review timing are often more flexible than base salary. Even £2-3k extra at the start compounds into £20k+ over 5 years."
    },
    {
        "title": "How to answer 'Tell me about yourself'",
        "url": "https://www.lennysnewsletter.com/p/tell-me-about-yourself",
        "source_type": "article", "topics": "interviews,hiring,careers",
        "content": "'Tell me about yourself' is the most common opener and the most poorly answered. The winning formula is Past-Present-Future in 90 seconds: (1) Past (20 seconds) — one sentence on your background and what led you here. (2) Present (40 seconds) — your most relevant recent experience, with one specific accomplishment that maps to this role. (3) Future (30 seconds) — why this specific role and company is the next logical step. Avoid: reciting your CV, mentioning personal life unless asked, or saying 'I'm a people person'. Practise it out loud until it sounds natural, not memorised."
    },
    {
        "title": "How to write a cover letter that actually gets read",
        "url": "https://www.lennysnewsletter.com/p/cover-letter-tips",
        "source_type": "article", "topics": "job-search,careers,hiring",
        "content": "Most cover letters are skimmed for 8 seconds or skipped entirely. To make yours one of the 20% that gets read: (1) Open with something specific — not 'I am writing to apply for...' but 'Your recent launch of X caught my attention because...'. (2) Three short paragraphs maximum. Para 1: why this company. Para 2: one concrete achievement that proves you can do this job. Para 3: a clear call to action. (3) Use the company's voice — if they're casual on their careers page, be casual. If formal, be formal. (4) Never reuse a generic letter — recruiters spot it instantly and bin it."
    },
    {
        "title": "How to optimise your LinkedIn profile for recruiters",
        "url": "https://www.lennysnewsletter.com/p/linkedin-profile",
        "source_type": "article", "topics": "job-search,careers,linkedin",
        "content": "LinkedIn is where recruiters search first. To appear in their searches: (1) Headline — don't just list your job title. Use 'Marketing Coordinator | Helping B2B SaaS scale through content and SEO' — keywords + value. (2) About section — 3 short paragraphs in first person, including the keywords for jobs you want. (3) Experience — every role should have 3-4 bullet points with quantified results, not job descriptions. (4) Turn on 'Open to Work' (privately, visible only to recruiters). (5) Post or comment once a week on industry topics — visibility drives inbound recruiter messages."
    },
    {
        "title": "How to switch careers without starting from scratch",
        "url": "https://www.lennysnewsletter.com/p/career-switching",
        "source_type": "article", "topics": "careers,job-search,career-change",
        "content": "Career switchers often undersell themselves by hiding their previous experience. The opposite works better: explicitly translate your old skills into the new field's language. A teacher moving into product management has stakeholder management, presentation skills, curriculum design (= roadmapping), and outcome measurement. A retail manager moving into operations has team leadership, scheduling, P&L responsibility. The framing for interviews is: 'Here is the problem you have. Here is how I've solved exactly this problem in a different context.' Don't apologise for your background — make it your edge."
    },
    {
        "title": "How to land remote-first roles",
        "url": "https://www.lennysnewsletter.com/p/remote-work",
        "source_type": "article", "topics": "remote-work,job-search,careers",
        "content": "Remote roles attract 5-10x more applicants than equivalent in-office roles. To stand out: (1) Demonstrate written communication on your application — a clear, well-structured cover letter signals you can work async. (2) Mention any prior remote experience explicitly, even informal — group projects, volunteer work, freelance. (3) Have a professional video setup for interviews — good lighting and audio matter more than the background. (4) Ask about their async culture — 'How does the team handle decisions across time zones?' shows you understand the realities of remote work, not just the perks."
    },
    {
        "title": "How to handle the 'What is your biggest weakness?' question",
        "url": "https://www.lennysnewsletter.com/p/interview-questions",
        "source_type": "article", "topics": "interviews,hiring,careers",
        "content": "The cliché 'I'm a perfectionist' answer is a red flag — it signals you can't self-reflect. A strong answer has three parts: (1) Name a real, non-disqualifying weakness — for a marketing role, 'public speaking to large groups' works; for a presenter role, it doesn't. (2) Show what you've done about it — 'I joined Toastmasters six months ago' or 'I took a course on X'. (3) Show measurable progress — 'I gave my first 50-person presentation last month and got positive feedback'. The point isn't the weakness — it's that you're self-aware and proactive about growth."
    },
    {
        "title": "Why side projects beat certifications for entry-level candidates",
        "url": "https://www.lennysnewsletter.com/p/side-projects",
        "source_type": "article", "topics": "careers,skills,job-search",
        "content": "Certifications prove you can pass a test. Side projects prove you can do the work. Hiring managers consistently prefer the latter. A 'side project' doesn't have to be elaborate — examples that have got people hired: a tear-down of a company's homepage UX with proposed improvements, a Substack with 6 posts analysing the industry you want to work in, a 5-minute Loom video presenting a strategy for a real product, a redesigned version of an app you use daily. Spend 20 hours on one substantive project rather than 100 hours on certifications. Then put it at the top of your CV."
    },
    {
        "title": "How to handle the salary expectations question",
        "url": "https://www.lennysnewsletter.com/p/salary-question",
        "source_type": "article", "topics": "interviews,compensation,negotiation",
        "content": "When asked early in the process 'What are your salary expectations?' — the worst answer is a specific number. Better responses: (1) Defer — 'I'd like to learn more about the role and team first. Can you share the budgeted range?'. (2) If pushed, give a wide range based on market research — Glassdoor, Levels.fyi, LinkedIn Salary, and Reddit's r/cscareerquestions all help. (3) If you must give a number, give your target +15% — you can always come down, never up. Never lie about your current salary — if asked, you can say 'I'd prefer to focus on the value of this new role rather than my current package'."
    },
    {
        "title": "How to use AI to massively accelerate your job search",
        "url": "https://www.lennysnewsletter.com/p/ai-job-search",
        "source_type": "podcast", "topics": "ai-tools,job-search,careers",
        "content": "AI tools cut job search time in half when used well. The workflow: (1) Use Claude or ChatGPT to extract the top 5 requirements from any job description. (2) Ask it to score your CV against those requirements out of 10 with reasons. (3) Have it rewrite your CV bullet points to mirror the job's language without lying. (4) Generate 10 likely interview questions from the JD. (5) Practise answers and have the AI roleplay as a tough interviewer. (6) Generate three thoughtful questions to ask them, based on the company's recent news. Spend 30 minutes per application using AI — quality beats quantity every time."
    },
    {
        "title": "How to recover from a bad interview answer",
        "url": "https://www.lennysnewsletter.com/p/interview-recovery",
        "source_type": "article", "topics": "interviews,hiring,careers",
        "content": "Everyone gives a bad answer at some point. What separates strong candidates is what they do next. Three recovery moves: (1) In the moment — pause, breathe, and say 'Let me think about that again' rather than rambling. Interviewers respect composure. (2) Later in the interview — circle back: 'I want to revisit my earlier answer on X — what I should have said was...'. This shows reflection and growth. (3) After the interview — in your thank-you email (always send one within 24 hours), briefly address the gap: 'On reflection, a stronger example for the question about Y would have been Z.' Most candidates don't do this — those who do stand out."
    },
    {
        "title": "Why your first job matters less than you think",
        "url": "https://www.lennysnewsletter.com/p/first-job",
        "source_type": "article", "topics": "careers,career-development",
        "content": "Junior candidates often agonise over the 'perfect' first job. The data shows this matters far less than people think. Three reasons not to wait for perfect: (1) Most people change jobs every 2-3 years early in their career — the first role is a stepping stone, not a destination. (2) Skills compound — a good manager and challenging work at a less-prestigious company beats a passive role at a famous one. (3) Money compounds — earning sooner means saving sooner, and the £15-20k you'd earn in 6 months of waiting is hard to make back. Take a good job now, deliver well for 18-24 months, then move strategically."
    },
    {
        "title": "How to ask for a referral without burning bridges",
        "url": "https://www.lennysnewsletter.com/p/referrals",
        "source_type": "article", "topics": "networking,job-search,careers",
        "content": "Employee referrals are 5-10x more likely to result in a hire than cold applications. How to ask without making it awkward: (1) Only ask people who actually know your work — a 1-hour coffee chat doesn't qualify. (2) Make it easy to say no — 'Would you be comfortable referring me for X role at Y? Completely understand if not.' Removing pressure makes 'yes' more likely. (3) Provide them with everything — your CV, a 3-line summary of why you're a fit, the exact job link. They should be able to forward your email in one click. (4) Always thank them publicly afterwards, regardless of outcome — it makes them want to help you again."
    },
    {
        "title": "How to handle multiple offers and competing deadlines",
        "url": "https://www.lennysnewsletter.com/p/multiple-offers",
        "source_type": "article", "topics": "careers,negotiation,job-search",
        "content": "Multiple offers are the strongest negotiating position you'll ever have. How to handle them ethically: (1) Be transparent — tell each company you're in late stages elsewhere. Most will speed up. (2) Never bluff about an offer you don't have — recruiters talk, and the industry is small. (3) Use the higher offer to negotiate the one you want more — 'I have an offer at X for £Y. I'd prefer to join you. Can you match or beat it?' Be specific. (4) Ask for time honestly — 'I have a decision deadline of Friday at the other role. Can you confirm by Thursday?' Most companies will accommodate this if asked respectfully."
    },
    {
        "title": "What good first 90 days look like in a new role",
        "url": "https://www.lennysnewsletter.com/p/first-90-days",
        "source_type": "article", "topics": "careers,career-development",
        "content": "Your first 90 days set the perception that lasts years. The structure that works: Days 1-30: Listen more than you talk. Meet everyone on your team and key stakeholders. Ask 'what's broken?' and 'what's working?'. Document everything. Days 31-60: Identify 1-2 quick wins that demonstrate competence without stepping on toes — improving a process, fixing a small bug, writing useful documentation. Days 61-90: Propose one meaningful project that addresses something you've identified. By day 90 you should have a clear scorecard with your manager: what does success look like at 6 months and 12 months?"
    },
    {
        "title": "How to spot a toxic workplace in an interview",
        "url": "https://www.lennysnewsletter.com/p/red-flags",
        "source_type": "article", "topics": "careers,hiring,job-search",
        "content": "Interviews go both ways. Red flags to watch for: (1) High turnover — ask 'how long has the team been together?' If most have been there under 12 months, that's a sign. (2) Vague answers about success metrics — if no one can tell you what 'good' looks like, the role lacks clarity. (3) Negative talk about the person you'd replace — if they trash their predecessor, they'll trash you. (4) Pressure to decide fast — exploding offers ('decide by Monday') often hide poor culture. (5) Glassdoor reviews with consistent themes — one bad review is an outlier, ten complaints about the same manager is a pattern."
    },
    {
        "title": "How to deal with imposter syndrome in a new role",
        "url": "https://www.lennysnewsletter.com/p/imposter-syndrome",
        "source_type": "article", "topics": "careers,career-development,mindset",
        "content": "Almost everyone feels like an imposter in a new role — especially high performers. What helps: (1) Keep a 'wins' document — a running list of things you've delivered, positive feedback, problems you've solved. Read it when you're doubting yourself. (2) Reframe 'I don't know' from weakness to honesty — asking good questions is a sign of competence, not incompetence. (3) Talk to one person at your level — they're usually feeling the same way, and saying it out loud removes 50% of the weight. (4) Remember the hiring process — many people evaluated you and chose you. Trust their judgment when yours is shaky."
    },
    {
        "title": "How to make the most of LinkedIn DMs",
        "url": "https://www.lennysnewsletter.com/p/cold-outreach",
        "source_type": "article", "topics": "networking,job-search,linkedin",
        "content": "Cold LinkedIn messages have a 20-30% reply rate when done well, and under 5% when done badly. The template that works: subject 'Quick question about [their specific role/project]'. Message: '[Personalised opener referencing something specific they posted or did]. I'm exploring a move into [field] and was struck by how you [specific thing]. Would you be open to 15 minutes to share what surprised you most about the transition? Happy to send specific questions in advance. Either way, thanks for the work you put out — it's been useful to me as I figure this out.' Three sentences. Specific. Low pressure. Genuine."
    },
    {
        "title": "What to do when your job search isn't working",
        "url": "https://www.lennysnewsletter.com/p/job-search-stuck",
        "source_type": "article", "topics": "job-search,careers,mindset",
        "content": "If you've applied to 50+ roles with no interviews, the problem is rarely 'the market'. Diagnose by elimination: (1) CV problem — are you getting through ATS? Test by applying to roles where you exceed every listed requirement. If still no callbacks, your CV needs work. (2) Targeting problem — are you applying to roles you're actually qualified for? Get a friend in the industry to honestly assess your fit. (3) Volume problem — 5 applications a week isn't enough in a tough market; 25 well-tailored applications is the floor. (4) Pipeline problem — are you only applying cold? Add networking and referrals. Fix one variable at a time, not all four at once."
    },
]


@app.route("/admin/seed-static", methods=["POST"])
@require_key
def admin_seed_static():
    """Embed and store curated career content directly — no scraping needed."""
    from pipeline.embedder import embed_and_store
    stored = embed_and_store(STATIC_CONTENT)
    return jsonify({"status": "done", "stored": stored, "total_content": len(STATIC_CONTENT)})


@app.route("/admin/scrape-url", methods=["POST"])
@require_key
def admin_scrape_url():
    """Scrape, embed and store a single URL. Fast — call once per URL."""
    import pipeline.scraper as _s
    _s.REQUEST_DELAY = 0
    from pipeline.scraper  import scrape_url, chunk_text
    from pipeline.embedder import embed_and_store
    d   = request.json or {}
    url = d.get("url", "")
    src = d.get("source_type", "article")
    topics = d.get("topics", ["general"])
    if not url:
        return jsonify({"error": "url required"}), 400
    doc = scrape_url(url, source_type=src, topics=topics)
    if not doc:
        return jsonify({"status": "skipped", "url": url, "reason": "paywalled or error"})
    parts  = chunk_text(doc["text"])
    chunks = [
        {"url": url, "title": doc["title"], "content": p,
         "source_type": src, "topics": ",".join(topics), "chunk_idx": i}
        for i, p in enumerate(parts)
    ]
    stored = embed_and_store(chunks)
    return jsonify({"status": "ok", "url": url, "chunks": stored, "title": doc["title"][:60]})


@app.route("/admin/seed", methods=["POST"])
@require_key
def admin_seed():
    """Start the seed pipeline in a background thread and return immediately."""
    import threading
    if _seed_state["status"] == "running":
        return jsonify({"status": "running", "log": _seed_state["log"][-10:]})
    if _seed_state["status"] == "done":
        return jsonify(_seed_state.get("result", {"status": "done"}))
    _seed_state["log"] = []
    t = threading.Thread(target=_run_seed, daemon=True)
    t.start()
    return jsonify({"status": "started", "message": "Seed running in background. Poll /admin/seed/status"})


@app.route("/admin/seed/status")
@require_key
def admin_seed_status():
    """Poll the seed pipeline progress."""
    return jsonify({
        "status": _seed_state["status"],
        "log":    _seed_state["log"][-20:],
        "result": _seed_state.get("result")
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
