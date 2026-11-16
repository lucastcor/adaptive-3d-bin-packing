import { Info } from "lucide-react";
import { useCallback, useRef, useState } from "react";

type InfoTipProps = {
  text: string;
};

type BubblePosition = {
  top: number;
  left: number;
  width: number;
  below: boolean;
};

/**
 * Tooltip com `position: fixed`: escapa de qualquer painel com overflow e é
 * sempre fixado (clamped) dentro da viewport — nunca é recortado.
 */
export function InfoTip({ text }: InfoTipProps) {
  const anchorRef = useRef<HTMLSpanElement | null>(null);
  const [position, setPosition] = useState<BubblePosition | null>(null);

  const show = useCallback(() => {
    const rect = anchorRef.current?.getBoundingClientRect();
    if (!rect) {
      return;
    }
    const margin = 10;
    const width = Math.min(290, window.innerWidth - margin * 2);
    let left = rect.left + rect.width / 2 - width / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - width - margin));
    const below = rect.top < 140;
    const top = below ? rect.bottom + 8 : rect.top - 8;
    setPosition({ top, left, width, below });
  }, []);

  const hide = useCallback(() => setPosition(null), []);

  return (
    <span
      ref={anchorRef}
      className="info-tip"
      tabIndex={0}
      aria-label={text}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
    >
      <Info size={13} aria-hidden="true" />
      {position ? (
        <span
          className="info-bubble-fixed"
          role="tooltip"
          data-below={position.below ? "true" : "false"}
          style={{ top: position.top, left: position.left, width: position.width }}
        >
          {text}
        </span>
      ) : null}
    </span>
  );
}
