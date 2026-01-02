'use client';

import { Script } from '@/types/episode';

interface ScriptTurnProps {
  script: Script;
  isActive: boolean;
  onClick: () => void;
}

export default function ScriptTurn({ script, isActive, onClick }: ScriptTurnProps) {
  return (
    <div
      className="flex flex-col gap-2 w-full cursor-pointer"
      onClick={onClick}
    >
      {/* Speaker tag */}
      <div
        className={`
          bg-white border border-solid rounded-[6px] px-3 py-[2px] inline-flex items-center justify-center w-fit
          ${isActive ? 'border-black' : 'border-[rgba(0,0,0,0.5)]'}
        `}
      >
        <span
          className={`
            font-regular text-[12px] text-center tracking-[-0.6px]
            ${isActive ? 'text-black' : 'text-[rgba(0,0,0,0.5)]'}
          `}
        >
          {script.speaker}
        </span>
      </div>

      {/* Script text */}
      <p
        className={`
          font-regular text-[16px] leading-normal
          ${isActive ? 'text-black' : 'text-[rgba(0,0,0,0.5)]'}
        `}
      >
        {script.text}
      </p>
    </div>
  );
}
