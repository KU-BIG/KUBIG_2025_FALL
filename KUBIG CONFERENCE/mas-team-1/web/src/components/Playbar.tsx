'use client';

import { useRef, useState, useEffect } from 'react';

interface PlaybarProps {
  audioSrc: string;
  currentTime: number;
  onTimeUpdate: (time: number) => void;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export default function Playbar({ audioSrc, currentTime, onTimeUpdate }: PlaybarProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const speedControlRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [internalTime, setInternalTime] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.1);
  const [isLooping, setIsLooping] = useState(false);
  const [isSpeedHovered, setIsSpeedHovered] = useState(false);
  const [isSpeedControlOpen, setIsSpeedControlOpen] = useState(false);

  // Close speed control when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (speedControlRef.current && !speedControlRef.current.contains(e.target as Node)) {
        setIsSpeedControlOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Sync external currentTime with audio
  useEffect(() => {
    if (audioRef.current && Math.abs(audioRef.current.currentTime - currentTime) > 0.5) {
      audioRef.current.currentTime = currentTime;
    }
  }, [currentTime]);

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setInternalTime(audioRef.current.currentTime);
      onTimeUpdate(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
      audioRef.current.playbackRate = playbackRate;
    }
  };

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      setInternalTime(time);
      onTimeUpdate(time);
    }
  };

  const skipBackward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - 10);
    }
  };

  const skipForward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.min(duration, audioRef.current.currentTime + 10);
    }
  };

  const updatePlaybackRate = (newRate: number) => {
    const clampedRate = Math.max(0.25, Math.min(2, newRate));
    const roundedRate = Math.round(clampedRate * 100) / 100;
    setPlaybackRate(roundedRate);
    if (audioRef.current) {
      audioRef.current.playbackRate = roundedRate;
    }
  };

  const incrementSpeed = () => updatePlaybackRate(playbackRate + 0.05);
  const decrementSpeed = () => updatePlaybackRate(playbackRate - 0.05);
  const resetSpeed = () => updatePlaybackRate(1);

  const toggleLoop = () => {
    setIsLooping(!isLooping);
    if (audioRef.current) {
      audioRef.current.loop = !isLooping;
    }
  };

  const progress = duration > 0 ? (internalTime / duration) * 100 : 0;

  return (
    <div className="bg-bg-primary h-[80px] w-full flex flex-col items-center justify-center gap-2">
      <audio
        ref={audioRef}
        src={audioSrc}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
      />

      {/* Controls */}
      <div className="flex items-center gap-4">
        {/* Playback speed */}
        <div ref={speedControlRef} className="relative">
          <button
            onClick={() => setIsSpeedControlOpen(!isSpeedControlOpen)}
            onMouseEnter={() => setIsSpeedHovered(true)}
            onMouseLeave={() => setIsSpeedHovered(false)}
            className="w-8 h-8 flex items-center justify-center relative"
          >
            <img src="/icons/gauge.svg" alt="speed" className="w-4 h-4 opacity-70" />

            {/* Hover tooltip - current speed */}
            {isSpeedHovered && !isSpeedControlOpen && (
              <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] font-medium text-black/70 whitespace-nowrap bg-white/80 px-1.5 py-0.5 rounded">
                {playbackRate.toFixed(2)}x
              </span>
            )}
          </button>

          {/* Speed control popover */}
          {isSpeedControlOpen && (
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-white rounded-xl shadow-lg border border-black/10 p-3 w-[140px]">
              <div className="flex items-center justify-between gap-2">
                {/* Decrement button */}
                <button
                  onClick={decrementSpeed}
                  disabled={playbackRate <= 0.25}
                  className="w-8 h-8 flex items-center justify-center rounded-lg border border-black/10 hover:bg-black/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <svg width="12" height="2" viewBox="0 0 12 2" fill="none">
                    <path d="M1 1H11" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </button>

                {/* Current speed display */}
                <div className="flex-1 text-center">
                  <span className="text-lg font-semibold tabular-nums">
                    {playbackRate.toFixed(2)}x
                  </span>
                </div>

                {/* Increment button */}
                <button
                  onClick={incrementSpeed}
                  disabled={playbackRate >= 2}
                  className="w-8 h-8 flex items-center justify-center rounded-lg border border-black/10 hover:bg-black/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path d="M6 1V11M1 6H11" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </button>
              </div>

              {/* Reset button */}
              <button
                onClick={resetSpeed}
                disabled={playbackRate === 1}
                className="w-full mt-2 py-1.5 text-xs text-black/60 hover:text-black hover:bg-black/5 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-1"
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M1 4.5C1.5 2.5 3.5 1 6 1C8.76 1 11 3.24 11 6C11 8.76 8.76 11 6 11C3.79 11 1.95 9.5 1.29 7.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  <path d="M1 1V4.5H4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                1.00x
              </button>
            </div>
          )}
        </div>

        {/* Skip backward */}
        <button
          onClick={skipBackward}
          className="w-8 h-8 flex items-center justify-center"
        >
          <img src="/icons/skip-back.svg" alt="skip back" className="w-4 h-4" />
        </button>

        {/* Play/Pause */}
        <button
          onClick={togglePlay}
          className="w-8 h-8 bg-black rounded-full flex items-center justify-center"
        >
          <img
            src={isPlaying ? '/icons/pause.svg' : '/icons/play.svg'}
            alt={isPlaying ? 'pause' : 'play'}
            className="w-4 h-4"
          />
        </button>

        {/* Skip forward */}
        <button
          onClick={skipForward}
          className="w-8 h-8 flex items-center justify-center"
        >
          <img src="/icons/skip-forward.svg" alt="skip forward" className="w-4 h-4" />
        </button>

        {/* Loop toggle */}
        <button
          onClick={toggleLoop}
          className={`w-8 h-8 flex items-center justify-center ${isLooping ? 'opacity-100' : 'opacity-50'}`}
        >
          <img src="/icons/repeat.svg" alt="repeat" className="w-4 h-4" />
        </button>
      </div>

      {/* Seekbar */}
      <div className="flex items-center gap-2 w-[538px]">
        <span className="font-bold text-[12px] text-[rgba(0,0,0,0.7)] w-[26px] text-right">
          {formatTime(internalTime)}
        </span>
        <div className="relative flex-1 h-3">
          <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-1 bg-[rgba(0,0,0,0.3)] rounded-[2px]" />
          <div
            className="absolute top-1/2 -translate-y-1/2 left-0 h-1 bg-[rgba(0,0,0,0.7)] rounded-[2px]"
            style={{ width: `${progress}%` }}
          />
          <input
            type="range"
            min={0}
            max={duration || 100}
            value={internalTime}
            onChange={handleSeek}
            className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer"
          />
        </div>
        <span className="font-bold text-[12px] text-[rgba(0,0,0,0.7)] w-[26px]">
          {formatTime(duration)}
        </span>
      </div>
    </div>
  );
}
