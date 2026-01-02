'use client';

import { motion } from 'framer-motion';
import { ArrowUp, ArrowDown, Minus } from 'lucide-react';
import type { TickerAnalysisSlide as TickerAnalysisSlideType } from '@/types/slide';
import { TradingViewWidget } from '../TradingViewWidget';

interface Props {
  slide: TickerAnalysisSlideType;
}

export function TickerAnalysisSlide({ slide }: Props) {
  const actionStyles = {
    BUY: { bg: 'bg-emerald-100', text: 'text-emerald-700', border: 'border-emerald-200', Icon: ArrowUp },
    HOLD: { bg: 'bg-amber-100', text: 'text-amber-700', border: 'border-amber-200', Icon: Minus },
    SELL: { bg: 'bg-red-100', text: 'text-red-700', border: 'border-red-200', Icon: ArrowDown },
  };

  const action = slide.action ? actionStyles[slide.action] : null;

  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
      <div className="px-8 py-8">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-start justify-between mb-6"
        >
          <div>
            <span className="inline-block px-3 py-1 bg-purple-100 text-purple-700 text-sm font-semibold rounded-lg mb-2">
              {slide.ticker}
            </span>
            <h2 className="text-2xl font-bold text-gray-900">{slide.title}</h2>
          </div>
          {action && (
            <div className={`flex items-center gap-2 px-4 py-2 rounded-xl ${action.bg} border ${action.border}`}>
              <action.Icon className={`w-5 h-5 ${action.text}`} />
              <span className={`text-sm font-bold ${action.text}`}>{slide.action}</span>
            </div>
          )}
        </motion.div>

        {slide.description && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="text-gray-600 leading-relaxed mb-6"
          >
            {slide.description}
          </motion.p>
        )}

        <div className="space-y-3 mb-6">
          {slide.points.map((point, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 + i * 0.1 }}
              className="flex items-start gap-3 bg-gray-50 rounded-xl p-4 border border-gray-100"
            >
              <span className="text-purple-600 font-bold text-lg">*</span>
              <p className="text-gray-800 leading-relaxed">{point}</p>
            </motion.div>
          ))}
        </div>

        {slide.charts && slide.charts.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="space-y-4"
          >
            {slide.charts.map((chart, i) => (
              <div key={i} className="rounded-xl overflow-hidden border border-gray-200 bg-white">
                {chart.title && (
                  <div className="px-4 py-3 border-b border-gray-100 bg-gray-50">
                    <span className="text-sm font-medium text-gray-600">{chart.title}</span>
                    <span className="ml-2 text-xs text-gray-400">({chart.ticker})</span>
                  </div>
                )}
                <TradingViewWidget symbol={chart.ticker} />
              </div>
            ))}
          </motion.div>
        )}
      </div>
    </div>
  );
}
