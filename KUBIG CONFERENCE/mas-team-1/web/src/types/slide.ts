export type SlideType =
  | 'title'
  | 'market-summary'
  | 'headline'
  | 'comparison'
  | 'stats'
  | 'ticker-intro'
  | 'ticker-analysis'
  | 'events'
  | 'closing';

export interface ChartSource {
  ticker: string;
  title?: string;
}

export interface BaseSlide {
  id: number;
  type: SlideType;
  turnId: number;
  charts?: ChartSource[];
}

export interface TitleSlide extends BaseSlide {
  type: 'title';
  date: string;
  nutshell: string;
  description?: string;
}

export interface MarketSummarySlide extends BaseSlide {
  type: 'market-summary';
  title?: string;
  description?: string;
  indices: Array<{
    name: string;
    value: number;
    change: number;
    changePercent: number;
  }>;
  commodities: Array<{
    name: string;
    value: number;
    change: number;
    changePercent: number;
  }>;
}

export interface HeadlineSlide extends BaseSlide {
  type: 'headline';
  icon?: string;
  title: string;
  subtitle: string;
  bullets?: string[];
  description?: string;
  theme?: 'red' | 'blue' | 'gold' | 'green' | 'purple';
}

export interface ComparisonSlide extends BaseSlide {
  type: 'comparison';
  title: string;
  description?: string;
  items: Array<{
    label: string;
    value: string;
    description: string;
    highlight?: boolean;
  }>;
}

export interface StatsSlide extends BaseSlide {
  type: 'stats';
  title: string;
  description?: string;
  stats: Array<{
    label: string;
    value: string;
    subtext?: string;
    trend?: 'up' | 'down' | 'neutral';
  }>;
  note?: string;
  theme?: 'red' | 'blue' | 'gold' | 'green' | 'purple';
}

export interface TickerIntroSlide extends BaseSlide {
  type: 'ticker-intro';
  ticker: string;
  companyName: string;
  currentPrice: number;
  dayChange: number;
  dayChangePercent: number;
  description?: string;
}

export interface TickerAnalysisSlide extends BaseSlide {
  type: 'ticker-analysis';
  ticker: string;
  title: string;
  points: string[];
  description?: string;
  action?: 'BUY' | 'HOLD' | 'SELL';
}


export interface EventsSlide extends BaseSlide {
  type: 'events';
  title: string;
  description?: string;
  events: Array<{
    date: string;
    label: string;
    description?: string;
  }>;
}

export interface ClosingSlide extends BaseSlide {
  type: 'closing';
  headline: string;
  tagline: string;
  description?: string;
}

export type Slide =
  | TitleSlide
  | MarketSummarySlide
  | HeadlineSlide
  | ComparisonSlide
  | StatsSlide
  | TickerIntroSlide
  | TickerAnalysisSlide
  | EventsSlide
  | ClosingSlide;
