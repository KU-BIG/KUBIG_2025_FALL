'use client';

import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { EpisodeListItem } from '@/types/episode';
import { formatDate } from '@/lib/format';
import TickerTag from './TickerTag';

const springTransition = {
  type: 'spring' as const,
  stiffness: 80,
  damping: 20,
  mass: 1,
};

interface EpisodeCardProps {
  episode: EpisodeListItem;
}

export default function EpisodeCard({ episode }: EpisodeCardProps) {
  const router = useRouter();

  const handleArrowClick = () => {
    // Clear back direction, use default forward animation
    delete document.documentElement.dataset.direction;

    const navigate = () => router.push(`/episode/${episode.date}`);

    // Use View Transitions API if available
    if (document.startViewTransition) {
      document.startViewTransition(navigate);
    } else {
      navigate();
    }
  };

  return (
    <div className="flex flex-col px-5 py-3 rounded-xl w-full group">
      {/* Date */}
      <p className="font-regular text-[16px] text-text-primary opacity-50">
        {formatDate(episode.date)}
      </p>

      {/* Title row with arrow - -2px gap from date */}
      <div className="flex items-center justify-between w-full mt-[-2px]">
        {/* Title with underline animation */}
        <div className="flex-1 pr-4 overflow-hidden flex items-center">
          <span className="relative inline-block max-w-full">
            <span className="font-medium text-[32px] text-text-primary leading-[48px] line-clamp-1">
              {episode.nutshell}
            </span>
            {/* Underline that draws from left on hover - matches text width */}
            <span className="absolute bottom-[2px] left-0 h-[1px] bg-text-primary w-full origin-left scale-x-0 group-hover:scale-x-100 transition-transform duration-[600ms] ease-out" />
          </span>
        </div>

        {/* Arrow button - only this is clickable */}
        <motion.button
          onClick={handleArrowClick}
          className="w-12 h-12 flex items-center justify-center flex-shrink-0 cursor-pointer"
          whileHover={{ x: -8 }}
          transition={springTransition}
        >
          <img
            src="/icons/arrow_right.svg"
            alt="arrow"
            className="w-12 h-12"
          />
        </motion.button>
      </div>

      {/* Ticker tags - 8px gap from title */}
      {episode.user_tickers.length > 0 && (
        <div className="flex gap-[6px] flex-wrap mt-[8px]">
          {episode.user_tickers.map((ticker) => (
            <TickerTag key={ticker} ticker={ticker} />
          ))}
        </div>
      )}
    </div>
  );
}
