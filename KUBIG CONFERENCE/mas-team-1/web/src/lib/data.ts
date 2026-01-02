import { Episode, EpisodeListItem } from '@/types/episode';
import { promises as fs } from 'fs';
import path from 'path';

export async function getEpisodeList(): Promise<EpisodeListItem[]> {
  const filePath = path.join(process.cwd(), 'public', 'data', 'episodes.json');
  const data = await fs.readFile(filePath, 'utf-8');
  return JSON.parse(data);
}

export async function getEpisode(date: string): Promise<Episode> {
  const filePath = path.join(process.cwd(), 'public', 'data', `${date}.json`);
  const data = await fs.readFile(filePath, 'utf-8');
  return JSON.parse(data);
}
