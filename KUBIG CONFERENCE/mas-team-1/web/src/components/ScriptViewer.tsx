'use client';

import { useRef, useEffect } from 'react';
import { Script } from '@/types/episode';
import ScriptTurn from './ScriptTurn';

interface ScriptViewerProps {
  scripts: Script[];
  currentTurnId: number;
  onTurnClick: (script: Script) => void;
}

export default function ScriptViewer({ scripts, currentTurnId, onTurnClick }: ScriptViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const turnRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  // Auto-scroll to current turn
  useEffect(() => {
    if (currentTurnId > 0) {
      const turnElement = turnRefs.current.get(currentTurnId);
      if (turnElement && containerRef.current) {
        const containerRect = containerRef.current.getBoundingClientRect();
        const turnRect = turnElement.getBoundingClientRect();

        // Calculate position to center the turn in the container
        const scrollTop = containerRef.current.scrollTop;
        const turnTop = turnRect.top - containerRect.top + scrollTop;
        const targetScroll = turnTop - containerRect.height / 2 + turnRect.height / 2;

        containerRef.current.scrollTo({
          top: targetScroll,
          behavior: 'smooth',
        });
      }
    }
  }, [currentTurnId]);

  return (
    <div
      ref={containerRef}
      className="flex flex-col gap-8 h-full overflow-y-auto overflow-x-hidden px-5"
    >
      {scripts.map((script) => (
        <div
          key={script.id}
          ref={(el) => {
            if (el) turnRefs.current.set(script.id, el);
          }}
        >
          <ScriptTurn
            script={script}
            isActive={script.id === currentTurnId}
            onClick={() => onTurnClick(script)}
          />
        </div>
      ))}
      {/* Bottom padding for scroll */}
      <div className="shrink-0 h-10" />
    </div>
  );
}
