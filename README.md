# Career Platform

  AI-powered career development tools for young people aged 16-30, powered by RAG (Retrieval Augmented Generation) using Lenny's Newsletter content.

  ## Three Tools

  1. **AI Career Coach** — Personalised career advice based on your age, qualifications, interests, and goals.
  2. **Mock Interviewer** — Practice 5 behavioural interview questions, scored on the STAR method with actionable coaching.
  3. **Employer Readiness Scorecard** — Gap analysis and a 30-day upskilling roadmap based on current employer demand.

  ## Stack

  - Python + Flask
  - Anthropic Claude (via Anthropic API)
  - OpenAI text-embedding-3-small (for RAG embeddings)
  - Supabase (PostgreSQL + pgvector)
  - BeautifulSoup4 (web scraping)

  ## Environment Variables

  ```bash
  ANTHROPIC_API_KEY=...
  OPENAI_API_KEY=...
  SUPABASE_URL=https://...
  SUPABASE_KEY=...
  RAG_API_KEY=...
  SESSION_SECRET=...
  ```

  ## Run Locally

  ```bash
  cd artifacts/flask-app
  pip install -r requirements.txt
  python3 main.py
  ```

  Visit http://localhost:5001

  ## Data Pipeline

  - `pipeline/scraper.py` — Scrape Lenny's free articles
  - `pipeline/embedder.py` — Create OpenAI embeddings and store in Supabase
  - `pipeline/run_full.py` — First-time seeding pipeline
  - `pipeline/refresh.py` — Incremental refresh of new articles
  - `pipeline/all_urls.py` — Master list of content URLs

  ## RAG System

  - `rag/retriever.py` — Semantic search over Supabase pgvector
  - Admin endpoint `POST /admin/seed-static` — Embed curated content without scraping

  ## Credits

  Built on insights from [Lenny's Newsletter](https://www.lennysnewsletter.com).
  