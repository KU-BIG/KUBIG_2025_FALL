'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { StatsSlide as StatsSlideType } from '@/types/slide';
import { TradingViewWidget } from '../TradingViewWidget';

interface Props {
  slide: StatsSlideType;
}

const themeStyles = {
  red: {
    bg: 'from-red-50 to-rose-50',
    border: 'border-red-100',
  },
  blue: {
    bg: 'from-blue-50 to-indigo-50',
    border: 'border-blue-100',
  },
  gold: {
    bg: 'from-amber-50 to-yellow-50',
    border: 'border-amber-100',
  },
  green: {
    bg: 'from-emerald-50 to-green-50',
    border: 'border-emerald-100',
  },
  purple: {
    bg: 'from-purple-50 to-violet-50',
    border: 'border-purple-100',
  },
};

export function StatsSlide({ slide }: Props) {
  const theme = themeStyles[slide.theme || 'blue'];

  const getTrendIcon = (trend?: 'up' | 'down' | 'neutral') => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-5 h-5 text-emerald-600" />;
      case 'down':
        return <TrendingDown className="w-5 h-5 text-red-600" />;
      default:
        return <Minus className="w-5 h-5 text-gray-400" />;
    }
  };

  return (
    <div className={`bg-gradient-to-br ${theme.bg} rounded-2xl border ${theme.border} shadow-sm overflow-hidden`}>
      <div className="px-8 py-8">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">{slide.title}</h2>
          {slide.description && (
            <p className="text-gray-600 leading-relaxed">{slide.description}</p>
          )}
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {slide.stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.1 }}
              className="bg-white rounded-xl p-5 border border-gray-100"
            >
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm text-gray-500">{stat.label}</p>
                {stat.trend && getTrendIcon(stat.trend)}
              </div>
              <p className="text-2xl font-bold text-gray-900 mb-1">{stat.value}</p>
              {stat.subtext && (
                <p className="text-sm text-gray-500">{stat.subtext}</p>
              )}
            </motion.div>
          ))}
        </div>

        {slide.note && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="text-sm text-gray-500 bg-white/50 rounded-lg p-4 border border-gray-100"
          >
            {slide.note}
          </motion.p>
        )}

        {slide.charts && slide.charts.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-6 space-y-4"
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
