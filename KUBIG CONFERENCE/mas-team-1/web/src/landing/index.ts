import type { Slide } from '@/types/slide';
import { slides as slides20251222 } from './20251222/slides';

const slidesMap: Record<string, Slide[]> = {
  '20251222': slides20251222,
};

export function getSlides(episodeDate: string): Slide[] {
  return slidesMap[episodeDate] || [];
}

export function hasSlides(episodeDate: string): boolean {
  return episodeDate in slidesMap;
}
