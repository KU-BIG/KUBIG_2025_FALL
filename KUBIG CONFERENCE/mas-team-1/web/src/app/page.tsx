import { getEpisodeList } from '@/lib/data';
import EpisodeCard from '@/components/EpisodeCard';

export default async function HomePage() {
  const episodes = await getEpisodeList();

  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Header Section */}
      <header className="px-10 pt-[67px] pb-0">
        <h1 className="font-title text-[64px] leading-[64px] tracking-[-3.2px] text-text-primary text-left">
          Yesterday&apos;s close, Today&apos;s edge
        </h1>
        <p className="font-extralight text-[24px] tracking-[-1.2px] text-text-primary mt-3 text-left ml-[10px]">
          매일 아침, AI가 전날 미국 증시의 주요 흐름과 핵심 종목 동향을 분석해 팟캐스트로 전달합니다.
          출근길이나 아침 루틴 중에 편하게 들으며 하루를 준비하세요.
        </p>
      </header>

      {/* Divider */}
      <div className="mx-10 mt-6 border-t border-border-default" />

      {/* Episode List */}
      <main className="px-10 py-5">
        <div className="flex flex-col gap-5">
          {episodes.map((episode) => (
            <EpisodeCard key={episode.date} episode={episode} />
          ))}
        </div>
      </main>
    </div>
  );
}
