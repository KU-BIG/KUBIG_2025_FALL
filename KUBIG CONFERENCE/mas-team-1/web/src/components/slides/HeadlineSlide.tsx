'use client';

import { motion } from 'framer-motion';
import { Film, Cpu, Coins, TrendingUp } from 'lucide-react';
import type { HeadlineSlide as HeadlineSlideType } from '@/types/slide';
import { TradingViewWidget } from '../TradingViewWidget';

interface Props {
  slide: HeadlineSlideType;
}

const themeStyles = {
  red: {
    bg: 'from-red-50 to-rose-50',
    border: 'border-red-100',
    accent: 'text-red-600',
    iconBg: 'bg-red-100',
    bulletBg: 'bg-red-50',
    bulletBorder: 'border-red-100',
  },
  blue: {
    bg: 'from-blue-50 to-indigo-50',
    border: 'border-blue-100',
    accent: 'text-blue-600',
    iconBg: 'bg-blue-100',
    bulletBg: 'bg-blue-50',
    bulletBorder: 'border-blue-100',
  },
  gold: {
    bg: 'from-amber-50 to-yellow-50',
    border: 'border-amber-100',
    accent: 'text-amber-600',
    iconBg: 'bg-amber-100',
    bulletBg: 'bg-amber-50',
    bulletBorder: 'border-amber-100',
  },
  green: {
    bg: 'from-emerald-50 to-green-50',
    border: 'border-emerald-100',
    accent: 'text-emerald-600',
    iconBg: 'bg-emerald-100',
    bulletBg: 'bg-emerald-50',
    bulletBorder: 'border-emerald-100',
  },
  purple: {
    bg: 'from-purple-50 to-violet-50',
    border: 'border-purple-100',
    accent: 'text-purple-600',
    iconBg: 'bg-purple-100',
    bulletBg: 'bg-purple-50',
    bulletBorder: 'border-purple-100',
  },
};

const icons: Record<string, React.ReactNode> = {
  film: <Film className="w-6 h-6" />,
  cpu: <Cpu className="w-6 h-6" />,
  coins: <Coins className="w-6 h-6" />,
  trending: <TrendingUp className="w-6 h-6" />,
};

export function HeadlineSlide({ slide }: Props) {
  const theme = themeStyles[slide.theme || 'blue'];

  return (
    <div className={`bg-gradient-to-br ${theme.bg} rounded-2xl border ${theme.border} shadow-sm overflow-hidden`}>
      <div className="px-8 py-8">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
          {slide.icon && icons[slide.icon] && (
            <div className={`inline-flex p-3 rounded-xl ${theme.iconBg} ${theme.accent} mb-4`}>
              {icons[slide.icon]}
            </div>
          )}
          <h2 className="text-2xl md:text-3xl font-bold text-gray-900 mb-2">{slide.title}</h2>
          <p className="text-lg text-gray-600">{slide.subtitle}</p>
        </motion.div>

        {slide.description && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="text-gray-700 leading-relaxed mb-6"
          >
            {slide.description}
          </motion.p>
        )}

        {slide.bullets && slide.bullets.length > 0 && (
          <div className="space-y-3 mb-6">
            {slide.bullets.map((bullet, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 + i * 0.1 }}
                className={`flex items-start gap-3 ${theme.bulletBg} border ${theme.bulletBorder} rounded-xl p-4`}
              >
                <span className={`${theme.accent} font-bold text-lg`}>*</span>
                <p className="text-gray-800 leading-relaxed">{bullet}</p>
              </motion.div>
            ))}
          </div>
        )}

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
