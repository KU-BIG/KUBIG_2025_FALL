# 미국 주식 장마감 AI 브리핑 서비스

http://43.201.213.158/
<br>
Team: `지피티야 팀명 추천해줘`

이 레포는 “장마감 브리핑 스크립트 생성 → TTS로 오디오 생성”까지의 파이프라인과, 그 입력이 되는 뉴스 데이터 수집용 AWS Lambda를 포함합니다.  
또한, 생성된 산출물을 재생/탐색하기 위한 Next.js 웹 플레이어(`web/`)를 포함합니다.

## 아키텍처 한눈에 보기

```mermaid
flowchart TD
  classDef ghost fill:#fff,stroke:#bbb,stroke-dasharray:3 3,color:#777;

  subgraph AWS["AWS: 뉴스 수집/저장"]
    direction LR
    AWS_EB["EventBridge Scheduler (30m)"] --> AWS_LAMBDA["Lambda (container)"]
    AWS_LAMBDA --> AWS_UPLOAD["upload_db: list crawl → DynamoDB (idempotent)"]
    AWS_LAMBDA --> AWS_DETAIL["detail_crawl: page crawl → S3(XML) + DynamoDB update"]
    AWS_UPLOAD --> AWS_DDB["DynamoDB: NEWS_TABLE"]
    AWS_DETAIL --> AWS_S3["S3: NEWS_BUCKET (article XML)"]
    AWS_DETAIL --> AWS_DDB
  end

  subgraph ENTRY["Entry + Prefetch (한 번만)"]
    direction LR
    IN["입력<br/>date(YYYYMMDD) + user_tickers(optional)"] --> ORCH["orchestrator.py<br/>LangGraph(StateGraph)"]
    ORCH --> PF["global_prefetch_node<br/>prefetch_all(date) → cache/{date}/...<br/>news_list.json, titles.txt, bodies/*<br/>calendar.csv, calendar.json<br/>market_context.json"]
  end

  PF --> OA

  subgraph AGENTS["Agents (스크립트 생성)"]
    direction LR
    OA["OpeningAgent"] --> TA["ThemeAgent"] --> CO["CompanyAgent<br/>(TickerPipeline: ticker별 분석 + 대본 생성)"] --> CA["ClosingAgent"]

    subgraph OAD["OpeningAgent 내부"]
      direction LR
      OA_LC["load_context_node"] --> OA_PM["prepare_initial_messages"] --> OA_A["agent (ReAct)"]
      OA_A -->|tool_calls| OA_T["ToolNode(TOOLS)"]
      OA_T --> OA_A
      OA_A -->|final| OA_EX["extract_script_node<br/>themes + nutshell + opening scripts"]
    end

    subgraph TAD["ThemeAgent 내부 (Fan-out / Fan-in)"]
      direction LR
      TA_LB["load_base_from_temp"] --> TA0["Fan-out: run_theme_workers<br/>batch(..., return_exceptions=True)"]

      subgraph TWPAR["ThemeWorkerGraph 병렬 실행 (테마별)"]
        direction LR

        subgraph W1["Worker#1 (theme[0])"]
          direction LR
          W1LC["load_context_node"] --> W1P["prepare_messages_node"] --> W1A["worker_agent_node (ReAct)"]
          W1A -->|tool_calls| W1T["ToolNode(TOOLS)"]
          W1T --> W1A
          W1A -->|final| W1E["extract_theme_scripts_node"]
        end

        subgraph W2["Worker#2 (theme[1])"]
          direction LR
          W2LC["load_context_node"] --> W2P["prepare_messages_node"] --> W2A["worker_agent_node (ReAct)"]
          W2A -->|tool_calls| W2T["ToolNode(TOOLS)"]
          W2T --> W2A
          W2A -->|final| W2E["extract_theme_scripts_node"]
        end

        subgraph W3["Worker#3 (theme[2])"]
          direction LR
          W3LC["load_context_node"] --> W3P["prepare_messages_node"] --> W3A["worker_agent_node (ReAct)"]
          W3A -->|tool_calls| W3T["ToolNode(TOOLS)"]
          W3T --> W3A
          W3A -->|final| W3E["extract_theme_scripts_node"]
        end

        WN["... Worker#N"]:::ghost
      end

      TA0 --> W1
      TA0 --> W2
      TA0 --> W3
      TA0 --> WN

      W1 --> TA1["Fan-in: merge_scripts<br/>opening + theme scripts"]
      W2 --> TA1
      W3 --> TA1
      WN --> TA1

      TA1 --> TA2["refine_transitions<br/>edits only"]
    end

    subgraph COD["CompanyAgent 내부 (Fan-out / Debate / Fan-in)"]
      direction LR
      CO0["Fan-out: tickers[]<br/>사용자 입력 ticker별 병렬 실행"]

      subgraph COPAR["CompanyWorker 병렬 실행 (ticker별)"]
        direction LR

        subgraph C1W["CompanyWorker#1 (ticker[0])"]
          direction LR
          C1LC["load_context<br/>news list + chart(30d 1d + today 5m) + SEC index"] --> C1R1["Round 1: Blind Analysis<br/>4 Persona 병렬"]
          C1R1 --> C1DEB["Rounds 2-N: Guided Debate<br/>Moderator 진행 (멀티턴)"]
          C1DEB --> C1CONS["Final Conclusion<br/>Consensus(action+confidence) + 근거 정리"]
          C1CONS --> C1SCR["Ticker Script Worker<br/>(tool-less, 6~8 turns)"]

          subgraph C1P["4 Expert Personas"]
            direction LR
            C1P1["Fundamental"]
            C1P2["Risk Manager"]
            C1P3["Growth Analyst"]
            C1P4["Sentiment"]
          end

          C1R1 --> C1P1
          C1R1 --> C1P2
          C1R1 --> C1P3
          C1R1 --> C1P4
          C1P1 --> C1DEB
          C1P2 --> C1DEB
          C1P3 --> C1DEB
          C1P4 --> C1DEB

          C1TOOLS["Tool calls (as needed)<br/>뉴스/차트/SEC 등 근거 조회"]:::ghost
          C1DEB -.-> C1TOOLS
        end

        subgraph C2W["CompanyWorker#2 (ticker[1])"]
          direction LR
          C2LC["load_context"] --> C2R1["Round 1: Blind Analysis"] --> C2DEB["Rounds 2-N: Guided Debate"] --> C2CONS["Final Conclusion"] --> C2SCR["Ticker Script Worker"]
        end

        subgraph C3W["CompanyWorker#3 (ticker[2])"]
          direction LR
          C3LC["load_context"] --> C3R1["Round 1: Blind Analysis"] --> C3DEB["Rounds 2-N: Guided Debate"] --> C3CONS["Final Conclusion"] --> C3SCR["Ticker Script Worker"]
        end

        CNW["... CompanyWorker#N"]:::ghost
      end

      CO0 --> C1W
      CO0 --> C2W
      CO0 --> C3W
      CO0 --> CNW

      C1SCR --> CO_M["Fan-in: merge_company_scripts<br/>기업별 파트 합치기 + id 정규화"]
      C2SCR --> CO_M
      C3SCR --> CO_M
      CNW --> CO_M

      CO_M --> CO_REF["ticker_script_refiner<br/>(tool-less, 전환부 다듬기)"]
      CO_REF --> CO_O["scripts[]에 append<br/>theme 뒤에 ticker 챕터 추가"]
    end

    subgraph CAD["ClosingAgent 내부"]
      direction LR
      CA_LS["load_scripts_from_temp"] --> CA_LC["load_context_node"] --> CA_PM["prepare_messages_node"] --> CA_A["agent (ReAct)"]
      CA_A -->|tool_calls| CA_T["ToolNode(TOOLS)<br/>get_calendar/get_ohlcv"]
      CA_T --> CA_A
      CA_A -->|final| CA_EX["extract_closing_turns_node"]
      CA_EX --> CA_APP["append_scripts_node"]
    end

    PF -.-> OA_LC
    PF -.-> W1LC
    PF -.-> CO0
    PF -.-> CA_LC

    OA --> OA_LC
    OA_EX --> TA

    TA --> TA_LB
    TA2 --> CO

    CO --> CO0
    CO_O --> CA

    CA --> CA_LS
  end

  subgraph ARTIFACTS["Artifacts"]
    direction LR
    TEMP_OPEN["temp/opening.json"] --> TEMP_THEME["temp/theme.json"] --> TEMP_TICKER["temp/ticker_pipeline.json"] --> TEMP_CLOSE["temp/closing.json"]
    TEMP_DEB["temp/debate/{date}/{TICKER}_debate.json<br/>(ticker별 Debate 산출물)"]
    POD["podcast/{date}/script.json<br/>(date/nutshell/user_tickers/chapter/scripts)"]
    DB1["podcast/podcast.db 업데이트<br/>upsert_script_row(...)"]
  end

  OA_EX --> TEMP_OPEN
  TA2 --> TEMP_THEME
  CO_O --> TEMP_TICKER
  CA_APP --> TEMP_CLOSE
  CO0 --> TEMP_DEB

  ORCH --> POD
  ORCH --> DB1

  subgraph TTS["TTS (script.json 생성 후 실행)"]
    direction LR
    POD --> TT0["python -m tts.src.tts<br/>(LangGraph)"]

    subgraph TTG["TTS Graph (turn-level)"]
      direction LR
      TT_CFG["load_config<br/>gemini_tts.yaml 로드/검증"] --> TT_V["validate_paths<br/>script.json 존재 확인"] --> TT_LS["load_script<br/>podcast/{date}/script.json 로드"]
      TT_LS --> TT_MAP["map_turns_with_chapter<br/>speaker→label + chapter 범위 적용"] --> TT_REQ["build_turn_requests<br/>instructions + text → prompt"]
      TT_REQ --> TF["generate_turn_audio_parallel<br/>turn별 TTS 병렬 생성(배치)"]

      subgraph TPAR["병렬 TTS 호출 (turn별)"]
        direction LR
        T0["turn#0 TTS → 000.wav"]
        T1["turn#1 TTS → 001.wav"]
        T2n["turn#2 TTS → 002.wav"]
        Tn["... turn#N TTS → NNN.wav"]:::ghost
      end

      TF --> T0
      TF --> T1
      TF --> T2n
      TF --> Tn

      T0 --> TT_TIM["compute_timeline<br/>프레임 기반 start/end(ms) 계산"]
      T1 --> TT_TIM
      T2n --> TT_TIM
      Tn --> TT_TIM

      TT_TIM --> TT_MERGE["merge_audio<br/>{date}.wav 생성"] --> TT_OUT["write_outputs<br/>timeline.json + {date}.json 저장 + DB 업데이트"]
    end

    TT0 --> TT_CFG
    TT_YAML["tts/config/gemini_tts.yaml"] --> TT_CFG

    GEM["Gemini TTS API<br/>GEMINI_API_KEY"]
    T0 -.-> GEM
    T1 -.-> GEM
    T2n -.-> GEM
    Tn -.-> GEM
  end

  subgraph TTS_OUT["TTS Outputs"]
    direction LR
    WAV["podcast/{date}/{date}.wav"]
    TURNS["podcast/{date}/tts/<turn>.wav"]
    TL["podcast/{date}/tts/timeline.json"]
    DATEJSON["podcast/{date}/{date}.json<br/>scripts[].time 주입"]
    DB2["podcast/podcast.db 업데이트<br/>update_tts_row(tts_done=true)"]
  end

  TT_OUT --> WAV
  TT_OUT --> TURNS
  TT_OUT --> TL
  TT_OUT --> DATEJSON
  TT_OUT --> DB2

  subgraph WEB["Web Frontend (Next.js)"]
    direction LR
    WEB_BUILD["npm run build:data<br/>(web/scripts/build-data.ts)"] --> WEB_PUB["web/public/{data,audio}"] --> WEB_APP["Next.js app (web/)"]
  end
  DB1 --> WEB_BUILD
  WAV --> WEB_BUILD
  DATEJSON --> WEB_BUILD

  %% Data dependencies
  AWS_DDB -. "prefetch_all/news" .-> PF
  AWS_S3 -. "get_news_content (tools)" .-> OA_T
```

### Orchestrator 캐시/산출물 시퀀스

```mermaid
sequenceDiagram
  participant CLI as "CLI (orchestrator.py)"
  participant PF as "global_prefetch_node"
  participant OA as "OpeningAgent"
  participant TA as "ThemeAgent"
  participant TP as "TickerPipeline"
  participant CA as "ClosingAgent"
  participant FS as "Filesystem"

  CLI->>PF: prefetch_all(date)
  PF->>FS: write cache/{date}/news_list.json, titles.txt
  PF->>FS: write cache/{date}/calendar.json, calendar.csv
  PF->>FS: write cache/{date}/market_context.json

  CLI->>OA: invoke (stage >= 0)
  OA->>FS: read cache/{date}/...
  OA->>FS: write temp/opening.json

  alt stage >= 1
    CLI->>TA: invoke
    TA->>FS: read cache/{date}/... (via tools)
    TA->>FS: write temp/theme.json
  end

  alt stage >= 2
    CLI->>TP: invoke
    TP->>FS: read cache/{date}/... (via tools)
    TP->>FS: write temp/debate/{date}/*_debate.json
    TP->>FS: write temp/ticker_pipeline.json
  end

  alt stage >= 3
    CLI->>CA: invoke
    CA->>FS: read cache/{date}/... (via tools)
    CA->>FS: write temp/closing.json
  end

  CLI->>FS: write podcast/{date}/script.json
  CLI->>FS: update podcast/podcast.db
  CLI->>FS: cleanup cache/{date} (graph + finalizer)
```

### Debate 실행 시퀀스 (per ticker)

```mermaid
sequenceDiagram
  participant CLI as "CLI (python -m debate.graph)"
  participant G as "agents/debate/graph.py"
  participant NL as "get_news_list"
  participant NC as "get_news_content"
  participant OH as "get_ohlcv"
  participant SL as "get_sec_filing_list"
  participant SC as "get_sec_filing_content"
  participant LLM as "LLM"

  CLI->>G: "run_debate(date,ticker,max_rounds)"
  G->>NL: "invoke(tickers=[ticker])"
  G->>SL: "invoke(forms=[10-K,10-Q])"
  G->>OH: "invoke(30d)"
  par "4 experts (parallel)"
    G->>LLM: "expert(system+user, tools)"
    LLM-->>NC: "tool_call get_news_content(...)"
    LLM-->>SC: "tool_call get_sec_filing_content(...)"
    LLM-->>OH: "tool_call get_ohlcv(...)"
    LLM-->>G: "final JSON(text,sources)"
  end
  G->>LLM: "moderator(rounds_json)"
  LLM-->>G: "JSON(needs_more_debate or conclusion)"
  G-->>CLI: "TickerDebateOutput JSON"
```

## 실행 (Quick Start)

### 0) 설정 파일

- **비밀키(API Key)**: `.env`에만 저장(권장)
  - 예: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `LANGSMITH_API_KEY`
- **비밀이 아닌 런타임 설정(모델/timeout/라운드/AWS 등)**: `config/app.yaml`로 관리(권장)
  - 로딩 우선순위: (shell export) > (`config/app.yaml`) > (`.env`, `override=False`)

시작:
1. `.env.example` → `.env` 복사 후 API 키 채우기
2. 필요하면 `config/app.yaml` 수정

### 1) 장마감 브리핑 스크립트 생성 (Opening → Theme → Ticker → Closing)

```bash
python orchestrator.py 20251222 -t GOOG AAPL
```

- 결과: `podcast/20251222/script.json`
- 참고: orchestrator는 실행 중 `cache/20251222/`를 만들고 종료 시 정리합니다(디버깅용 산출물은 `temp/`와 `podcast/`에 남음).

### 2) TTS 실행 (turn 단위 오디오 생성 + 합본)

```bash
python -m tts.src.tts 20251222
```

- 입력: `podcast/20251222/script.json`
- 출력:
  - `podcast/20251222/tts/*.wav` (turn별)
  - `podcast/20251222/tts/timeline.json`
  - `podcast/20251222/20251222.wav` (합본)
  - `podcast/20251222/20251222.json` (time 주입된 최종 스크립트)

### 3) Web 플레이어 실행 (Next.js)

```bash
cd web
npm install

# DB에서 데이터 빌드 + 개발 서버 실행
npm run dev:fresh
```

- 접속: `http://localhost:3000`
- 참고: `podcast/podcast.db`가 갱신된 뒤에는 `npm run build:data`가 필요합니다.

## 스크립트 파이프라인 상세

상위 오케스트레이터: `orchestrator.py` (문서: `ORCHESTRATOR.md`)

### 단계 구성

- **global_prefetch**
  - `cache/{date}/news_list.json`, `calendar.csv`, `market_context.json` 등을 미리 생성
- **OpeningAgent** (`agents/opening/graph.py`)
  - `nutshell`(한 줄 요약), `themes`(테마 후보), 오프닝 대본 생성 → `temp/opening.json`
- **ThemeAgent** (`agents/theme/graph.py`)
  - 테마별 Worker 병렬 실행(fan-out) → 병합(fan-in) → 전환 Refiner → `temp/theme.json`
- **TickerPipeline** (`orchestrator.py:ticker_pipeline_node`)
  - 사용자 티커(`-t/--tickers`) 기준
  - fan-out: 티커별 Debate → `temp/debate/{date}/{TICKER}_debate.json`
  - fan-out: 티커별 Script Worker(tool-less) → fan-in merge → Refiner(tool-less) → `temp/ticker_pipeline.json`
- **ClosingAgent** (`agents/closing/graph.py`)
  - 누적 대본 입력으로 마무리 파트 생성 → `temp/closing.json`

### 최종 산출물(`podcast/{date}/script.json`) 구조(요약)

```json
{
  "date": "20251222",
  "nutshell": "string",
  "user_tickers": ["GOOG"],
  "chapter": [
    { "name": "opening", "start_id": 0, "end_id": 5 },
    { "name": "theme", "start_id": 6, "end_id": 25 },
    { "name": "ticker", "start_id": 26, "end_id": 40 },
    { "name": "closing", "start_id": 41, "end_id": 45 }
  ],
  "scripts": [
    { "id": 0, "speaker": "진행자", "text": "…", "sources": [] }
  ]
}
```

## TTS 파이프라인 상세

엔트리포인트: `tts/src/tts.py` (문서: `tts/ARCHITECTURE.md`)

- 입력: `podcast/{date}/script.json`
- 설정: `tts/config/gemini_tts.yaml` (speaker별 instruction/voice/timeout/병렬도 등)
- 필수 환경변수: `GEMINI_API_KEY`

## AWS Lambda (뉴스 수집 파이프라인)

문서: `LAMBDA.md`, 코드: `Lambda/`, 이미지: `Lambda.Dockerfile`

- 트리거: EventBridge 스케줄러 `kubig-LambdaTrigger` (약 30분 주기)
- 기능:
  - Yahoo Finance Latest News(US) 크롤링 → DynamoDB에 메타데이터 적재(멱등)
  - 상세 기사 크롤링 → XML 직렬화 후 S3 저장 → DynamoDB에 `path/publish_et_iso/provider/related_articles` 업데이트
- 환경변수(대표):
  - `TABLE_NAME` (DynamoDB)
  - `BUCKET_NAME` (S3)
  - `AWS_REGION`

이 Lambda가 채운 DynamoDB/S3 데이터는 스크립트 파이프라인의 뉴스 툴(`shared/tools/news.py`) 및 프리페치(`shared/fetchers/news.py`)에서 사용됩니다.

## Web Frontend

문서: `WEB.md`, 코드: `web/`

### Quick Start

```bash
cd web

# 의존성 설치
npm install

# DB에서 데이터 빌드 + 개발 서버 실행
npm run dev:fresh

# 또는 개발 서버만 실행 (기존 데이터 사용)
npm run dev
```

### Scripts

| 명령어 | 설명 |
|--------|------|
| `npm run dev` | 개발 서버 실행 (localhost만) |
| `npm run dev:network` | 개발 서버 실행 (외부 네트워크 접근 허용, `내IP:3000`) |
| `npm run dev:fresh` | 데이터 빌드 + 개발 서버 |
| `npm run build:data` | DB → public/ 데이터 빌드만 |
| `npm run build` | 데이터 빌드 + 프로덕션 빌드 |
| `npm run start` | 프로덕션 서버 실행 |

### 데이터 흐름

```
../podcast/podcast.db          # SQLite DB (에피소드 메타데이터)
../podcast/{date}/{date}.json  # 에피소드 스크립트 데이터
../podcast/{date}/{date}.wav   # 에피소드 오디오 파일
        ↓
  npm run build:data (scripts/build-data.ts)
        ↓
public/data/episodes.json      # 에피소드 목록
public/data/{date}.json        # 에피소드 상세 데이터
public/audio/{date}.wav        # 오디오 파일
```

## 주요 디렉토리

```text
agents/            # Opening/Theme/Closing (+ Debate는 agents/debate)
debate/            # Debate/Types wrapper + ticker_script 파이프라인(티커 대본)
shared/            # tools/fetchers/config/utils (공용)
config/            # app.yaml (비밀 아닌 런타임 설정)
podcast/           # 최종 산출물 + DB
tts/               # TTS 파이프라인
Lambda/            # 뉴스 수집 AWS Lambda
web/               # Next.js 웹 플레이어
```

## 참고 문서

- `ORCHESTRATOR.md`
- `agents/opening/ARCHITECTURE.md`
- `agents/theme/ARCHITECTURE.md`
- `agents/closing/ARCHITECTURE.md`
- `debate/ARCHITECTURE.md`
- `podcast/ARCHITECTURE.md`
- `tts/ARCHITECTURE.md`
- `LAMBDA.md`
- `WEB.md`
