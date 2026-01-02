'use client';

import { motion } from 'framer-motion';
import type { ComparisonSlide as ComparisonSlideType } from '@/types/slide';
import { TradingViewWidget } from '../TradingViewWidget';

interface Props {
  slide: ComparisonSlideType;
}

export function ComparisonSlide({ slide }: Props) {
  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
      <div className="px-8 py-8">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">{slide.title}</h2>
          {slide.description && (
            <p className="text-gray-600 leading-relaxed">{slide.description}</p>
          )}
        </motion.div>

        <div className="space-y-4 mb-6">
          {slide.items.map((item, i) => (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.15 }}
              className={`rounded-xl p-5 ${
                item.highlight
                  ? 'bg-emerald-50 border-2 border-emerald-200'
                  : 'bg-gray-50 border border-gray-100'
              }`}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-lg font-semibold text-gray-700">{item.label}</span>
                <span
                  className={`text-xl font-bold ${
                    item.highlight ? 'text-emerald-600' : 'text-gray-900'
                  }`}
                >
                  {item.value}
                </span>
              </div>
              <p className="text-gray-600 leading-relaxed">{item.description}</p>
            </motion.div>
          ))}
        </div>

        {slide.charts && slide.charts.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
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
