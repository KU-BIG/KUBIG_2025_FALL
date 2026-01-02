import type { Slide } from '@/types/slide';

export const slides: Slide[] = [
  // Opening
  {
    id: 0,
    type: 'title',
    turnId: 0,
    date: '2025-12-22',
    nutshell: 'AI와 미디어 빅딜이 이끄는 선별 랠리',
    description:
      '연준 완화 기대와 금 랠리가 배경이 되는 가운데, AI 인프라 투자와 할리우드 M&A가 시장을 주도하고 있습니다. 오늘 미국 주식 장마감 브리핑에서는 주요 시장 동향과 함께 LLY(Eli Lilly) 종목 분석을 다룹니다.',
  },
  {
    id: 1,
    type: 'market-summary',
    turnId: 1,
    title: '오늘의 시장 요약',
    description:
      '미국 주요 지수가 3거래일 연속 상승세를 이어갔습니다. 연준의 금리 인하 기대감과 AI 관련 기술주 강세가 시장을 견인했습니다.',
    indices: [
      { name: 'S&P 500', value: 6834.50, change: 40.73, changePercent: 0.60 },
      { name: 'NASDAQ', value: 23307.62, change: 116.54, changePercent: 0.50 },
      { name: 'DOW', value: 48134.89, change: 240.67, changePercent: 0.50 },
    ],
    commodities: [
      { name: 'Gold', value: 4440, change: 80, changePercent: 1.83 },
      { name: 'Silver', value: 68.50, change: 1.65, changePercent: 2.47 },
    ],
  },

  // Hollywood Big Deal (Turn 6-13)
  {
    id: 2,
    type: 'headline',
    turnId: 6,
    icon: 'film',
    title: 'Hollywood Big Deal',
    subtitle: 'WBD 인수전 전면전',
    description:
      '파라마운트 스카이댄스가 Warner Bros Discovery에 대한 적대적 인수를 제안하면서 할리우드 미디어 업계에 대형 M&A 경쟁이 본격화되고 있습니다. 래리 엘리슨이 $404억 규모의 개인 보증을 제공하며 거래의 신뢰성을 높였습니다.',
    bullets: [
      '파라마운트 스카이댄스의 적대적 인수 제안 - 회사 전체 대상',
      '래리 엘리슨 $404억 개인 보증으로 자금 조달 확보',
      '넷플릭스 vs 파라마운트 경쟁 구도 형성',
      'WBD 주가 +3.5% 상승, $27.77 → $28.75',
    ],
    theme: 'red',
  },
  {
    id: 3,
    type: 'headline',
    turnId: 8,
    title: 'Warner Bros Discovery (WBD)',
    subtitle: '인수 기대감에 상승세 지속',
    description:
      'WBD 주가는 인수 경쟁 소식에 힘입어 12월 들어 16% 이상 상승했습니다. 인수 제안가가 현재 주가 대비 프리미엄을 제공하고 있어 추가 상승 여력이 있다는 분석입니다.',
    theme: 'red',
    charts: [{ ticker: 'WBD', title: 'Warner Bros Discovery' }],
  },
  {
    id: 4,
    type: 'comparison',
    turnId: 10,
    title: 'WBD 인수 제안 비교',
    description:
      '두 거대 미디어 기업이 WBD 인수를 위해 경쟁하고 있습니다. 넷플릭스는 스트리밍 자산에 집중한 반면, 파라마운트는 회사 전체를 대상으로 더 공격적인 제안을 내놓았습니다.',
    items: [
      {
        label: 'Netflix',
        value: '$800억',
        description: '스튜디오+스트리밍 자산만 대상, 주당 약 $27 수준, 선택적 인수 전략',
        highlight: false,
      },
      {
        label: 'Paramount-Skydance',
        value: '$1,080억',
        description: '회사 전체 대상, 주당 $30 현금 제안, 래리 엘리슨 개인보증 포함',
        highlight: true,
      },
    ],
  },
  {
    id: 5,
    type: 'stats',
    turnId: 12,
    title: '2025년 M&A 붐',
    description:
      '올해 미국 M&A 시장은 역대급 호황을 보이고 있습니다. 금리 인하 기대감과 기업들의 성장 전략이 맞물리면서 대형 딜이 줄을 잇고 있습니다.',
    stats: [
      { label: '12월 발표 거래', value: '$4,600억+', subtext: '월간 기준 사상 최대', trend: 'up' },
      { label: '전년 대비', value: '+30%', subtext: '거래 규모 증가', trend: 'up' },
      { label: '연간 예상', value: '$4.8조', subtext: '역대 2위 기록', trend: 'up' },
    ],
    note: '연준 금리 인하와 규제 완화 기대가 M&A 시장 활성화의 주요 요인입니다.',
    theme: 'red',
  },

  // AI Infrastructure (Turn 14-21)
  {
    id: 6,
    type: 'headline',
    turnId: 14,
    icon: 'cpu',
    title: 'AI Infrastructure',
    subtitle: '인프라 투자 열풍',
    description:
      'AI 인프라에 대한 투자가 가속화되고 있습니다. NVIDIA의 중국 공급 재개와 Alphabet의 대규모 인수가 AI 반도체와 데이터센터 시장의 성장을 보여주고 있습니다.',
    bullets: [
      'NVIDIA H200 중국 공급 2월 재개 예정 - 초기 4~8만개 칩',
      'Alphabet $47.5억에 인터섹트 인수 - 데이터센터 역량 강화',
      '글로벌 테크 기업 회사채 발행 사상 최대 - $4,200억',
      'AI 관련 기술주 선별 랠리 지속',
    ],
    theme: 'blue',
  },
  {
    id: 7,
    type: 'headline',
    turnId: 15,
    title: 'NVIDIA (NVDA)',
    subtitle: 'H200 중국 공급 기대감',
    description:
      'NVIDIA 주가는 중국 시장 재진입 기대감에 1.5% 상승했습니다. H200은 수출 규제를 충족하면서도 고성능을 제공하는 제품으로, 중국 클라우드 업체들의 수요가 예상됩니다.',
    theme: 'blue',
    charts: [{ ticker: 'NVDA', title: 'NVIDIA' }],
  },
  {
    id: 8,
    type: 'stats',
    turnId: 17,
    title: 'NVIDIA H200 중국 공급 상세',
    description:
      'NVIDIA가 수출 규제를 충족하는 H200 칩을 중국에 공급할 예정입니다. 이는 중국 AI 시장에서의 점유율 유지를 위한 전략적 결정입니다.',
    stats: [
      { label: '공급 시기', value: '2월 중순', subtext: '춘절 연휴 전 배송 목표', trend: 'neutral' },
      { label: '초기 공급량', value: '4~8만개', subtext: '칩 기준, 점진적 확대 예정', trend: 'neutral' },
      { label: '관세', value: '25%', subtext: '미 행정부 부과 조건', trend: 'down' },
    ],
    note: '수출 규제 준수를 위해 성능을 조정한 버전이며, 향후 공급량은 수요에 따라 확대될 전망입니다.',
    theme: 'blue',
  },
  {
    id: 9,
    type: 'stats',
    turnId: 19,
    title: 'Alphabet 인터섹트 인수',
    description:
      'Alphabet이 데이터센터 인프라 기업 인터섹트를 $47.5억에 인수합니다. 이는 AI 워크로드 처리를 위한 컴퓨팅 인프라 확보 전략의 일환입니다.',
    stats: [
      { label: '인수가', value: '$47.5억', subtext: '현금 거래', trend: 'neutral' },
      { label: '주요 자산', value: '데이터센터', subtext: '+ 발전 프로젝트 포함', trend: 'up' },
      { label: '전력 확보', value: '수 GW 규모', subtext: 'AI 인프라 운영용', trend: 'up' },
    ],
    note: 'AI 모델 훈련과 추론을 위한 대규모 컴퓨팅 인프라 수요가 이런 인수를 촉진하고 있습니다.',
    theme: 'blue',
  },
  {
    id: 10,
    type: 'stats',
    turnId: 21,
    title: '글로벌 테크 회사채 발행',
    description:
      '2025년 글로벌 기술 기업들의 회사채 발행이 사상 최대를 기록했습니다. 대부분이 AI 인프라 투자에 사용될 예정입니다.',
    stats: [
      { label: '2025년 발행', value: '$4,200억', subtext: '사상 최대 규모', trend: 'up' },
      { label: '미국 기술주', value: '$3,400억+', subtext: '전체의 80% 차지', trend: 'up' },
      { label: '주요 용도', value: 'AI 설비', subtext: '데이터센터, GPU 확보', trend: 'neutral' },
    ],
    note: '레버리지 비율이 빠르게 상승하고 있어 향후 금리 변동에 대한 리스크 관리가 필요합니다.',
    theme: 'purple',
  },

  // Gold Rally (Turn 22-29)
  {
    id: 11,
    type: 'headline',
    turnId: 22,
    icon: 'coins',
    title: 'Gold & Silver Rally',
    subtitle: '1979년 이후 최고의 해',
    description:
      '금 가격이 $4,440으로 사상 최고치를 경신했습니다. 연준의 금리 인하 기대와 지정학적 리스크 고조가 안전자산 선호를 강화하고 있습니다.',
    bullets: [
      '금 $4,440 사상 최고 - YTD +70% 상승',
      '은 $68.50 - YTD +130% 상승',
      '연준 추가 금리 인하 기대 (2026년 2회 예상)',
      '지정학적 리스크 고조로 안전자산 수요 증가',
    ],
    theme: 'gold',
  },
  {
    id: 12,
    type: 'stats',
    turnId: 25,
    title: '금 랠리 배경 분석',
    description:
      '금 가격 상승은 단순한 투기가 아닌 구조적 요인에 기반합니다. 중앙은행들의 금 보유량 확대와 실질금리 하락이 핵심 동력입니다.',
    stats: [
      { label: 'YTD 수익률', value: '+70%', subtext: '1979년 이후 최고', trend: 'up' },
      { label: '연준 금리', value: '-0.75%p', subtext: '3회 인하 완료', trend: 'down' },
      { label: '2026년 전망', value: '2회 추가', subtext: '인하 기대', trend: 'down' },
    ],
    note: '중앙은행들의 달러 의존도 축소와 인플레이션 헤지 수요가 금 가격을 지지합니다.',
    theme: 'gold',
  },
  {
    id: 13,
    type: 'stats',
    turnId: 27,
    title: '위험/안전자산 동시 강세',
    description:
      '주식과 금이 동시에 상승하는 이례적인 상황입니다. 풍부한 유동성과 분산 투자 수요가 이 현상을 설명합니다.',
    stats: [
      { label: 'S&P 500', value: '+0.6%', subtext: '3일 연속 상승', trend: 'up' },
      { label: 'Gold', value: '+1.8%', subtext: '사상 최고가', trend: 'up' },
      { label: 'Silver', value: '+130%', subtext: 'YTD 수익률', trend: 'up' },
    ],
    note: '유동성이 풍부한 환경에서 주식과 금 모두 자금이 유입되고 있습니다. 이는 투자자들의 분산 전략을 반영합니다.',
    theme: 'gold',
  },

  // LLY Analysis (Turn 30-37)
  {
    id: 14,
    type: 'ticker-intro',
    turnId: 30,
    ticker: 'LLY',
    companyName: 'Eli Lilly and Company',
    currentPrice: 1076.11,
    dayChange: -0.61,
    dayChangePercent: -0.06,
    description:
      'Eli Lilly는 GLP-1 비만치료제 시장의 선두주자로, Mounjaro와 Zepbound를 통해 폭발적인 매출 성장을 기록하고 있습니다. 오늘 종목 분석에서는 성장 잠재력과 밸류에이션 리스크를 함께 살펴봅니다.',
  },
  {
    id: 15,
    type: 'ticker-analysis',
    turnId: 31,
    ticker: 'LLY',
    title: '주가 동향 분석',
    description:
      'LLY 주가는 최고점 대비 조정을 받았지만 여전히 높은 밸류에이션을 유지하고 있습니다. 최근 1개월간 박스권 움직임을 보이며 방향성을 모색 중입니다.',
    points: [
      '현재가 $1,076 - 고가권 숨고르기 구간',
      '최근 1개월 박스권(±2%) 움직임',
      '기관 중심 홀딩 기조 유지',
      '단기 방향성보다 중장기 관점 필요',
    ],
    charts: [{ ticker: 'LLY', title: 'Eli Lilly' }],
  },
  {
    id: 16,
    type: 'ticker-analysis',
    turnId: 33,
    ticker: 'LLY',
    title: 'GLP-1 비만치료제 중심 성장',
    description:
      'Eli Lilly의 핵심 성장 동력은 GLP-1 계열 비만치료제입니다. Mounjaro와 Zepbound의 매출이 빠르게 증가하고 있으며, 대규모 생산시설 투자가 진행 중입니다.',
    points: [
      'Mounjaro, Zepbound 매출 급성장 - 분기별 40%+ 성장률',
      '2024년 CAPEX $50억+ 투자 - 생산능력 확대',
      '총부채 $425억 (레버리지 확대 중)',
      'GLP-1 플랫폼에 집중된 성장 - 파이프라인 다변화 필요',
    ],
  },
  {
    id: 17,
    type: 'comparison',
    turnId: 35,
    title: '2030년 밸류에이션 시나리오',
    description:
      '현재 주가 $1,076 기준으로 2030년 목표주가 시나리오를 분석했습니다. 성장률과 멀티플에 따라 큰 폭의 차이가 발생합니다.',
    items: [
      {
        label: '보수적',
        value: '$540-660',
        description: 'EPS $30, PER 18-22x 적용. 경쟁 심화, 약가 규제 시나리오. 현재가 대비 -40% 하락 리스크.',
        highlight: false,
      },
      {
        label: '기준',
        value: '$1,100-1,300',
        description: 'EPS $50, PER 22-26x 적용. 현재 성장률 유지 시나리오. 현재가 대비 소폭 상승.',
        highlight: false,
      },
      {
        label: '낙관적',
        value: '$1,700+',
        description: 'EPS $65, PER 26-32x 적용. 시장 점유율 확대, 신약 성공 시나리오.',
        highlight: false,
      },
    ],
  },
  {
    id: 18,
    type: 'ticker-analysis',
    turnId: 37,
    ticker: 'LLY',
    title: '리스크 분석',
    description:
      '현재 주가에는 낙관적 시나리오가 상당 부분 반영되어 있어 안전마진이 부족합니다. 주요 리스크 요인을 고려할 때 신규 매수보다는 관망을 권고합니다.',
    points: [
      '약가 규제 리스크 - IRA, 340B 프로그램 확대 가능성',
      'GLP-1 안전성 이슈 - 장기 부작용 데이터 축적 필요',
      '경쟁 심화 - Novo Nordisk, 후발 주자 진입',
      '현 가격 안전마진 부족 - PER 60x 이상 프리미엄',
    ],
    action: 'SELL',
  },

  // Closing (Turn 38-45)
  {
    id: 19,
    type: 'headline',
    turnId: 38,
    title: '오늘의 핵심',
    subtitle: '연준 완화 + 선별 랠리',
    description:
      '오늘 시장의 핵심은 연준의 통화 완화 기대와 AI/미디어 섹터의 선별적 강세입니다. 금 가격 사상 최고가 경신은 불확실성 속에서도 자산 분산의 중요성을 보여줍니다.',
    bullets: [
      'AI/미디어 빅딜이 시장 주도 - WBD, NVDA 강세',
      '금 $4,440 사상 최고 - 안전자산 수요 지속',
      '내년 상반기 경기/물가 변수 - 연준 정책 방향 주목',
    ],
    theme: 'blue',
  },
  {
    id: 20,
    type: 'events',
    turnId: 41,
    title: '주요 일정',
    description:
      '향후 2주간 시장에 영향을 줄 주요 경제 지표와 이벤트입니다. 특히 GDP 성장률과 Core PCE 데이터는 연준 정책 방향에 중요한 힌트를 제공할 것입니다.',
    events: [
      {
        date: '12/23',
        label: 'GDP 성장률, Core PCE',
        description: '성장과 물가의 조합 - 소프트랜딩 시나리오 점검',
      },
      {
        date: '12/30',
        label: 'FOMC 의사록 공개',
        description: '12월 회의 논의 내용 - 금리 인하 경로 힌트',
      },
      {
        date: '01/05',
        label: 'ISM 제조업 PMI',
        description: '경기 바닥 신호 확인 - 반등 가능성 점검',
      },
      {
        date: '01/09',
        label: '비농업 고용',
        description: '고용 시장 강도 점검 - 연준 정책에 영향',
      },
    ],
  },
  {
    id: 21,
    type: 'closing',
    turnId: 44,
    headline: '미국 주식 장마감 브리핑',
    tagline: "Yesterday's close, Today's edge",
    description:
      '오늘도 미국 주식 장마감 브리핑을 시청해주셔서 감사합니다. 내일은 연말 결산 특집으로 2025년 시장을 돌아보고 2026년 전망을 다룰 예정입니다. 좋은 하루 되세요!',
  },
];
