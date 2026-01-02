'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';
import type { TickerIntroSlide as TickerIntroSlideType } from '@/types/slide';
import { TradingViewWidget } from '../TradingViewWidget';

interface Props {
  slide: TickerIntroSlideType;
}

export function TickerIntroSlide({ slide }: Props) {
  const isPositive = slide.dayChangePercent >= 0;

  return (
    <div className="bg-gradient-to-br from-purple-50 via-white to-indigo-50 rounded-2xl border border-purple-100 shadow-sm overflow-hidden">
      <div className="px-8 py-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center mb-8"
        >
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4"
          >
            <span className="inline-block px-4 py-2 bg-purple-100 text-purple-700 text-2xl font-bold rounded-xl mb-2">
              {slide.ticker}
            </span>
            <p className="text-lg text-gray-600">{slide.companyName}</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mb-4"
          >
            <p className="text-5xl font-bold text-gray-900">
              ${slide.currentPrice.toLocaleString()}
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex items-center justify-center gap-3"
          >
            {isPositive ? (
              <div className="flex items-center gap-2 px-4 py-2 bg-emerald-50 rounded-xl">
                <TrendingUp className="w-5 h-5 text-emerald-600" />
                <span className="text-lg font-bold text-emerald-600">
                  +{slide.dayChangePercent.toFixed(2)}%
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-2 px-4 py-2 bg-red-50 rounded-xl">
                <TrendingDown className="w-5 h-5 text-red-600" />
                <span className="text-lg font-bold text-red-600">
                  {slide.dayChangePercent.toFixed(2)}%
                </span>
              </div>
            )}
            <span className="text-gray-500">
              ({isPositive ? '+' : ''}${slide.dayChange.toFixed(2)})
            </span>
          </motion.div>
        </motion.div>

        {slide.description && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="text-gray-700 leading-relaxed text-center max-w-2xl mx-auto mb-8"
          >
            {slide.description}
          </motion.p>
        )}

        {slide.charts && slide.charts.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
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
