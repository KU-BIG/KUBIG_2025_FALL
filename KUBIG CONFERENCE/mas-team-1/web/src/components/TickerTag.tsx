interface TickerTagProps {
  ticker: string;
}

export default function TickerTag({ ticker }: TickerTagProps) {
  return (
    <div className="bg-bg-card border border-border-default rounded-[6px] px-3 py-[2px] inline-flex items-center justify-center">
      <span className="font-regular text-[12px] text-text-primary text-center tracking-[-0.6px]">
        {ticker}
      </span>
    </div>
  );
}
