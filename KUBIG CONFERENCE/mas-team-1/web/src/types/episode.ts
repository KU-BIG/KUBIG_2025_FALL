export interface Episode {
  date: string;
  nutshell: string;
  user_tickers: string[];
  chapter: Chapter[];
  scripts: Script[];
}

export interface Chapter {
  name: 'opening' | 'theme' | 'ticker' | 'closing';
  start_id: number;
  end_id: number;
}

export interface Script {
  id: number;
  speaker: '진행자' | '해설자';
  text: string;
  sources: Source[];
  time: [number, number]; // [start_ms, end_ms]
}

export interface Source {
  type: 'chart' | 'article' | 'event' | 'sec_filing';
  ticker?: string;
  start_date?: string;
  end_date?: string;
  pk?: string;
  title?: string;
  id?: string;
  date?: string;
  form?: string;
  filed_date?: string;
  accession_number?: string;
}

export interface EpisodeListItem {
  date: string;
  nutshell: string;
  user_tickers: string[];
}
