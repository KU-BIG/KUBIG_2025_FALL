'use client';

import { motion } from 'framer-motion';
import type { TitleSlide as TitleSlideType } from '@/types/slide';

interface Props {
  slide: TitleSlideType;
}

export function TitleSlide({ slide }: Props) {
  const formattedDate = new Date(slide.date).toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  return (
    <div className="bg-gradient-to-br from-amber-50 via-white to-orange-50 rounded-2xl border border-amber-100 shadow-sm overflow-hidden">
      <div className="px-8 py-12 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-amber-100 text-amber-700 rounded-full text-sm font-medium mb-6">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            {formattedDate}
          </div>

          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            <span className="bg-gradient-to-r from-amber-500 via-orange-500 to-amber-500 bg-clip-text text-transparent">
              미국 주식 장마감 브리핑
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-gray-800 font-medium leading-relaxed mb-6">
            {slide.nutshell}
          </p>

          {slide.description && (
            <p className="text-gray-600 leading-relaxed max-w-2xl mx-auto">
              {slide.description}
            </p>
          )}
        </motion.div>
      </div>
    </div>
  );
}
