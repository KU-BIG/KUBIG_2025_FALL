# 🧭 PATHFINDER: 나만의 커리어 탐색 에이전트

#### KUBIG 2025 Conference Project
#### Team Curator : 김수환, 남수빈, 백서현, 성용빈, 윤채영
> #### 불확실한 직무 방향성으로 고민하는 취업 준비생을 위한 Multi-Agent 기반의 맞춤형 커리어 솔루션

![Project Banner](https://img.shields.io/badge/KUBIG-Data%20Science-red?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge) ![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange?style=for-the-badge)

## 📖 Introduction

**"매일 쏟아지는 채용 공고 속에서, 나에게 진짜 맞는 직무는 무엇일까?"**

PATHFINDER는 단순한 키워드 검색을 넘어, 사용자의 경험과 가치관을 심층적으로 이해하고 이를 실제 채용 시장의 데이터와 연결해주는 **LLM 기반 멀티 에이전트 시스템**입니다.

### 💡 Why Agent?
- **비정형 데이터 연결:** 사용자의 모호한 경험(User Experience)과 비정형 채용 공고(Job Posting)를 LLM의 추론 능력으로 연결합니다.
- **능동적 탐색:** 사용자가 검색하는 것이 아니라, 에이전트가 먼저 질문하고(Interview), 평가하고(Critic), 추천(Recommend)합니다.
- **실질적 수행:** 직무 추천에서 끝나는 것이 아니라, 실제 공고를 찾아주고 일정 관리 도구(Notion/Google Calendar)에 저장하는 Action까지 수행합니다.

---

## 🚀 System Architecture & Key Features

이 프로젝트는 **LangGraph**를 기반으로 유기적으로 연결된 4개의 전문 에이전트로 구성되어 있습니다.

### 1️⃣ Personal Interview Agent (개인 인터뷰)
사용자와의 대화를 통해 **P-E Fit(Person-Environment Fit)** 이론에 기반한 페르소나를 구축합니다.
- **P-J / P-O / P-R Fit:** 직무 역량, 조직 문화, 보상 등 3가지 축으로 정보를 추출합니다.
- **NCS 기반 가설 검증:** 사용자의 답변에서 부족한 역량을 추론하여(Ontological Approach), NCS 데이터베이스를 기반으로 "혹시 이런 경험은 없으신가요?"라고 역으로 질문하여 잠재 역량을 이끌어냅니다.
- **Schema Extraction:** 대화 내용을 구조화된 JSON 데이터(Profile)로 실시간 업데이트합니다.

### 2️⃣ NCS Job Recommendation Agent (직무 추천)
구축된 페르소나를 바탕으로 NCS(국가직무능력표준) 상의 표준 직무를 매칭합니다.
- **Self-Reflective Search:** `Search` → `Critic` → `Refine` 루프를 통해 추천 결과가 사용자 프로필과 특정 점수 이상 일치할 때까지 검색 전략을 스스로 수정합니다.
- **Terminology Translation:** 딱딱한 행정 용어인 NCS 직무명을 채용 시장에서 사용하는 트렌디한 직무명(예: 응용SW엔지니어링 → 백엔드 개발자)으로 변환합니다.

### 3️⃣ Posting Recommendation Agent (공고 추천)
추천된 직무 카테고리에 맞춰 실제 채용 사이트(사람인 등)의 공고를 매칭합니다.
- **2-Step Filtering:**
 1. **SQL Hard Filtering:** 경력(User 경력 + 2년까지 유연 적용), 학력, 기술 스택 기반 1차 필터링.
  2. **Vector Similarity & AI Review:** 임베딩 유사도 분석 후, LLM이 공고의 맥락을 읽고 `상/중/하` 등급과 추천 사유를 생성합니다.
- **Customized Reason:** 왜 이 공고가 사용자에게 적합한지 구체적인 이유를 제공합니다.

### 4️⃣ Posting Management Agent (공고 관리)
최종 선택한 공고를 사용자의 생산성 도구와 연동합니다.
- **Notion Integration:** 채용 공고 데이터베이스에 기업명, 마감일, 링크 등을 체계적으로 저장합니다.
- **Google Calendar Integration:** 지원 마감일을 캘린더에 자동 등록하고 알림을 설정합니다.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Core** | Python 3.9+, LangChain, LangGraph |
| **LLM & Embedding** | OpenAI GPT-4o, text-embedding-3-small |
| **Vector DB** | FAISS |
| **Database** | SQLite |
| **Integrations** | Notion API, Google Calendar API |
| **Data Source** | Saramin API, Custom Crawler |
| **UI** | Chainlit |

---

## 📂 Project Structure

```bash
path_finder_ku_2025/
│
├── 📄 chainlit_app.py                  # 메인 애플리케이션 (스테이지 관리)
├── 📄 chainlit.md                      # 환영 페이지
├── 📄 .env                             # 환경 변수 (API Keys)
├── 📄 requirements.txt                 # Python 패키지 의존성
├── 📄 README.md                        # 프로젝트 문서
│
├── 📂 .chainlit/
│   └── config.toml                     # Chainlit 설정 (테마, 타임아웃 등)
│
├── 📂 ui/                              # UI 래퍼 레이어 (Chainlit Step 시각화)
│   ├── __init__.py
│   ├── interview_ui.py                 # Interview Agent UI
│   ├── ncs_job_recommender_ui.py       # NCS Job Recommender Agent UI
│   ├── posting_recommender_ui.py       # Posting Recommender Agent UI
│   └── posting_manager_ui.py           # Posting Manager UI
│
├── 📂 agents/                          # 🤖 에이전트 모듈 폴더
│   ├── __init__.py
│   ├── interview_agent.py              # 인터뷰 진행 및 프로필 생성
│   ├── ncs_job_recommender_agent.py    # # NCS 기반 직무 추천
│   ├── posting_recommender_agent.py    # 채용 공고 검색 및 매칭
│   └── posting_manager_agent.py        # 공고 저장 및 관리
│
├── tools/                              # 🛠️ 유틸리티 도구 모음
│   ├── user_transformer.py             # 사용자 데이터 변환 및 임베딩
│   ├── data_filter.py                  # SQL 기반 데이터 필터링
│   ├── vector_engine.py                # 벡터 유사도 검색 엔진
│   ├── llm_reviewer.py                 # 공고 적합성 리뷰어
│   ├── ncs_recommend.py                # NCS 카테고리 코드 매핑
│   ├── notion_tool.py                  # Notion API 래퍼
│   ├── google_calendar_tool.py         # Google Calendar API 래퍼
│   ├── credentials.json                # Google OAuth 자격증명 (**직접 추가 필요**)
│   └── token.json                      # Google OAuth 토큰 (자동 생성)
│
├── 📂 core/
│   ├── __init__.py
│   └── state.py                        # MainState TypedDict 정의
│
└── 📂 data/                            # 데이터베이스 및 벡터 인덱스
    ├── job_service.db                  # SQLite: jobs, job_roles 테이블
    ├── ncs_vectorstore/                # FAISS: NCS 직무 벡터 검색
    │   ├── index.faiss
    │   └── index.pkl
    └── ncs_faiss_index/                # FAISS: 인터뷰 가설 생성용
        ├── index.faiss
        └── index.pkl

```

---

## 🏃‍♂️ Usage

Chainlit을 사용하여 웹 인터페이스로 에이전트와 대화할 수 있습니다.


### 1️⃣ 필수 요구사항
- **Python**: 3.8 이상
- **운영체제**: Windows / macOS / Linux
- **필수 계정**:
    - OpenAI API Key (GPT-4o, GPT-4o-mini)
    - Notion Integration Token
    - Google Cloud Project Token

### 2️⃣ 저장소 클론

```bash
git clone https://github.com/HaeAnn0203/path_finder_ku_2025.git
cd path_finder_ku_2025
git checkout Chainlit  # Chainlit 브랜치로 전환
```

### 3️⃣ Python 패키지 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 4️⃣ 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 입력하세요.

```ini
# OpenAI API
OPENAI_API_KEY=sk-proj-...

# Notion API (선택)
NOTION_API_KEY=secret_...
NOTION_JOB_DB_ID=...

# 데이터베이스 경로
DB_PATH=data/job_service.db
```

| 변수명 | 설명 |
| :--- | :--- |
| `OPENAI_API_KEY` | OpenAI API 키 (GPT-4o-mini 사용) |
| `NOTION_API_KEY` | Notion 저장 시 필요 |
| `NOTION_JOB_DB_ID` | Notion Database ID |
| `DB_PATH` | SQLite DB 경로 |

### 5️⃣ Notion 설정


1. [Notion My Integrations](https://www.notion.so/my-integrations) 접속 후 "New integration" 생성.
2. 발급된 `Internal Integration Token`을 `.env`의 `NOTION_API_KEY`에 입력.
3. Notion에서 새 데이터베이스 생성 후 아래 속성 추가:
    - `공고명`(Title), `회사명`(Text), `경력`(Text), `링크`(URL), `마감일`(Date), `지원상태`(Select)
4. 데이터베이스 페이지 우측 상단 `...` -> `Add connections` -> 생성한 Integration 연결.
5. 데이터베이스 URL에서 ID 추출하여 `.env`의 `NOTION_JOB_DB_ID`에 입력.


### 6️⃣ Google Calendar 설정


1. [Google Cloud Console](https://console.cloud.google.com/) 접속 및 새 프로젝트 생성.
2. "Google Calendar API" 검색 후 활성화.
3. `사용자 인증 정보 만들기` -> `OAuth 2.0 클라이언트 ID` (데스크톱 앱).
4. JSON 다운로드 후 `post_manager/tools/credentials.json` 경로에 저장.
5. 최초 실행 시 브라우저 인증 진행.

---

## 🚀 실행 방법

### Chainlit Web UI 실행

```bash
# 기본 실행
chainlit run chainlit_app.py

# Watch 모드 (코드 수정 시 자동 재시작)
chainlit run chainlit_app.py -w
```

---

## 📺 Demo Video

[![PathFinder Demo Video](http://img.youtube.com/vi/o9kplttEQVg/maxresdefault.jpg)](https://youtu.be/o9kplttEQVg)
