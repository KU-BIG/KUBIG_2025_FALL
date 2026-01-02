'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { Episode, Script } from '@/types/episode';
import { formatDateKorean } from '@/lib/format';
import ScriptViewer from './ScriptViewer';
import Playbar from './Playbar';
import { Slideshow } from './Slideshow';

const springTransition = {
  type: 'spring' as const,
  stiffness: 80,
  damping: 20,
  mass: 1,
};

interface EpisodePlayerProps {
  episode: Episode;
}

export default function EpisodePlayer({ episode }: EpisodePlayerProps) {
  const router = useRouter();
  const [currentTime, setCurrentTime] = useState(0);

  // Find current turn based on audio time
  const getCurrentTurnId = useCallback((timeInSeconds: number): number => {
    const timeInMs = timeInSeconds * 1000;
    for (let i = episode.scripts.length - 1; i >= 0; i--) {
      const script = episode.scripts[i];
      if (timeInMs >= script.time[0]) {
        return script.id;
      }
    }
    return 0;
  }, [episode.scripts]);

  const currentTurnId = getCurrentTurnId(currentTime);

  const handleTimeUpdate = (time: number) => {
    setCurrentTime(time);
  };

  const handleTurnClick = (script: Script) => {
    // Convert ms to seconds and update time
    const timeInSeconds = script.time[0] / 1000;
    setCurrentTime(timeInSeconds);
  };

  const handleSlideClick = (turnId: number) => {
    // Find the script with matching turnId
    const script = episode.scripts.find((s) => s.id === turnId);
    if (script) {
      const timeInSeconds = script.time[0] / 1000;
      setCurrentTime(timeInSeconds);
    }
  };

  const handleBackClick = (e: React.MouseEvent) => {
    e.preventDefault();
    document.documentElement.dataset.direction = 'back';

    const navigate = () => router.push('/');

    // Use View Transitions API if available
    if (document.startViewTransition) {
      document.startViewTransition(navigate);
    } else {
      navigate();
    }
  };

  const audioSrc = `/audio/${episode.date}.wav`;

  return (
    <div className="h-screen bg-bg-primary flex flex-col overflow-hidden">
      {/* Header - 80px */}
      <header className="h-[80px] px-10 flex items-center relative shrink-0">
        {/* Back button */}
        <a
          href="/"
          onClick={handleBackClick}
          className="flex items-center gap-0 cursor-pointer absolute left-10 group"
        >
          <motion.img
            src="/icons/arrow_left.svg"
            alt="back"
            className="w-8 h-8"
            whileHover={{ x: 4 }}
            transition={springTransition}
          />
          <span className="font-bold text-[16px] text-black">목록</span>
        </a>

        {/* Date - centered */}
        <div className="absolute left-1/2 -translate-x-1/2">
          <span className="font-bold text-[24px] text-black">
            {formatDateKorean(episode.date)}
          </span>
        </div>
      </header>

      {/* Main content area - height: calc(100vh - 160px) for header + playbar */}
      <main className="flex-1 flex gap-5 px-10 min-h-0 overflow-hidden">
        {/* Landing page - left side (4 columns, hidden on small screens) */}
        <div className="hidden lg:flex lg:flex-[4] bg-zinc-950 rounded-xl min-w-0 overflow-hidden">
          <Slideshow
            currentTurnId={currentTurnId}
            episodeDate={episode.date}
            onSlideClick={handleSlideClick}
          />
        </div>

        {/* Script viewer - right side (2 columns, full width on small screens) */}
        <div className="flex-1 lg:flex-[2] bg-white rounded-xl overflow-hidden flex flex-col min-w-0">
          <div className="pt-10 pb-10 flex-1 min-h-0 overflow-hidden">
            <ScriptViewer
              scripts={episode.scripts}
              currentTurnId={currentTurnId}
              onTurnClick={handleTurnClick}
            />
          </div>
        </div>
      </main>

      {/* Playbar - 80px */}
      <div className="shrink-0">
        <Playbar
          audioSrc={audioSrc}
          currentTime={currentTime}
          onTimeUpdate={handleTimeUpdate}
        />
      </div>
    </div>
  );
}
