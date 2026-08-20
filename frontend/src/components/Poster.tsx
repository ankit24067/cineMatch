interface Props {
  title: string;
  poster: string | null;
  className?: string;
}

export function Poster({ title, poster, className = "" }: Props) {
  if (!poster) {
    return (
      <div
        className={`flex items-center justify-center bg-[#1b1e26] p-3 text-center text-[11px] leading-tight text-neutral-400 ${className}`}
      >
        <span className="line-clamp-4">{title}</span>
      </div>
    );
  }

  return (
    <img
      src={poster}
      alt={`${title} poster`}
      loading="lazy"
      className={`object-cover ${className}`}
    />
  );
}
