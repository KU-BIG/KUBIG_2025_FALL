'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown } from 'lucide-react';
import type { MarketSummarySlide as MarketSummarySlideType } from '@/types/slide';

interface Props {
  slide: MarketSummarySlideType;
}

export function MarketSummarySlide({ slide }: Props) {
  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
      <div className="px-8 py-8">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            {slide.title || 'Market Overview'}
          </h2>
          {slide.description && (
            <p className="text-gray-600 leading-relaxed">{slide.description}</p>
          )}
        </motion.div>

        <div className="mb-8">
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
            Major Indices
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {slide.indices.map((idx, i) => (
              <motion.div
                key={idx.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="bg-gray-50 rounded-xl p-5 border border-gray-100"
              >
                <p className="text-sm text-gray-500 mb-1">{idx.name}</p>
                <p className="text-2xl font-bold text-gray-900 mb-2">
                  {idx.value.toLocaleString()}
                </p>
                <div className="flex items-center gap-2">
                  {idx.changePercent >= 0 ? (
                    <div className="flex items-center gap-1 px-2 py-1 bg-emerald-50 rounded-lg">
                      <TrendingUp className="w-4 h-4 text-emerald-600" />
                      <span className="text-sm font-semibold text-emerald-600">
                        +{idx.changePercent.toFixed(2)}%
                      </span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-1 px-2 py-1 bg-red-50 rounded-lg">
                      <TrendingDown className="w-4 h-4 text-red-600" />
                      <span className="text-sm font-semibold text-red-600">
                        {idx.changePercent.toFixed(2)}%
                      </span>
                    </div>
                  )}
                  <span className="text-sm text-gray-400">
                    {idx.changePercent >= 0 ? '+' : ''}
                    {idx.change.toFixed(2)}
                  </span>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
            Commodities
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {slide.commodities.map((com, i) => (
              <motion.div
                key={com.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + i * 0.1 }}
                className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl p-5 border border-amber-100"
              >
                <p className="text-sm text-amber-700 font-medium mb-1">{com.name}</p>
                <p className="text-2xl font-bold text-gray-900 mb-2">
                  ${com.value.toLocaleString()}
                </p>
                <div className="flex items-center gap-1 px-2 py-1 bg-emerald-50 rounded-lg w-fit">
                  <TrendingUp className="w-4 h-4 text-emerald-600" />
                  <span className="text-sm font-semibold text-emerald-600">
                    +{com.changePercent.toFixed(2)}%
                  </span>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
