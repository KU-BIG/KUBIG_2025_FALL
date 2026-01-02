'use client';

import { motion } from 'framer-motion';
import type { ClosingSlide as ClosingSlideType } from '@/types/slide';

interface Props {
  slide: ClosingSlideType;
}

export function ClosingSlide({ slide }: Props) {
  return (
    <div className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 rounded-2xl shadow-lg overflow-hidden">
      <div className="px-8 py-16 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-amber-400 via-yellow-300 to-amber-400 bg-clip-text text-transparent">
              {slide.headline}
            </span>
          </h2>
          <p className="text-xl text-gray-400 font-light mb-8">{slide.tagline}</p>

          {slide.description && (
            <p className="text-gray-300 leading-relaxed max-w-xl mx-auto">
              {slide.description}
            </p>
          )}
        </motion.div>
      </div>
    </div>
  );
}
