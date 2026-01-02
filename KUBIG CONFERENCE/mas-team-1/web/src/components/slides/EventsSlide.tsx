'use client';

import { motion } from 'framer-motion';
import { Calendar } from 'lucide-react';
import type { EventsSlide as EventsSlideType } from '@/types/slide';

interface Props {
  slide: EventsSlideType;
}

export function EventsSlide({ slide }: Props) {
  return (
    <div className="bg-gradient-to-br from-blue-50 via-white to-indigo-50 rounded-2xl border border-blue-100 shadow-sm overflow-hidden">
      <div className="px-8 py-8">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center gap-3 mb-6"
        >
          <div className="p-3 bg-blue-100 rounded-xl">
            <Calendar className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{slide.title}</h2>
            {slide.description && (
              <p className="text-gray-600">{slide.description}</p>
            )}
          </div>
        </motion.div>

        <div className="space-y-4">
          {slide.events.map((event, i) => (
            <motion.div
              key={event.label}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex items-start gap-4 bg-white rounded-xl p-5 border border-gray-100"
            >
              <span className="flex-shrink-0 px-3 py-2 bg-blue-100 text-blue-700 font-mono text-sm font-semibold rounded-lg">
                {event.date}
              </span>
              <div className="flex-1">
                <p className="text-lg font-semibold text-gray-900 mb-1">{event.label}</p>
                {event.description && (
                  <p className="text-gray-600">{event.description}</p>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
