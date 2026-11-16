import { Activity, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { fetchSystemUsage } from "../lib/api";
import type { SystemUsage } from "../types";

type FloatingMonitorProps = {
  onClose: () => void;
};

function formatGiB(bytes: number | null | undefined) {
  if (!bytes || bytes <= 0) {
    return "—";
  }
  return `${(bytes / 1024 ** 3).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} GiB`;
}

function level(percent: number) {
  return percent > 85 ? "high" : percent > 60 ? "mid" : "low";
}

/**
 * Monitor de recursos flutuante: acompanha CPU/memória de qualquer aba, sem
 * sair da tela atual (útil durante campanhas e execuções paralelas).
 */
export function FloatingMonitor({ onClose }: FloatingMonitorProps) {
  const [usage, setUsage] = useState<SystemUsage | null>(null);
  const historyRef = useRef<number[]>([]);
  const [history, setHistory] = useState<number[]>([]);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const next = await fetchSystemUsage();
        if (!active) {
          return;
        }
        setUsage(next);
        if (next.available && typeof next.cpu_percent === "number") {
          historyRef.current = [...historyRef.current, next.cpu_percent].slice(-40);
          setHistory(historyRef.current);
        }
      } catch {
        if (active) {
          setUsage({ available: false, reason: "sem conexão" });
        }
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 2000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const cpu = usage?.cpu_percent ?? 0;
  const memory = usage?.memory_percent ?? 0;
  const sparkPath =
    history.length >= 2
      ? history
          .map((value, index) => {
            const x = (index / (history.length - 1)) * 100;
            const y = 30 - (Math.min(value, 100) / 100) * 30;
            return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
          })
          .join(" ")
      : null;

  return (
    <aside className="floating-monitor" role="complementary" aria-label="Monitor de recursos">
      <div className="floating-monitor-head">
        <span>
          <Activity size={14} aria-hidden="true" /> Recursos
        </span>
        <button type="button" onClick={onClose} aria-label="Fechar monitor" title="Fechar (reabra no topo da tela)">
          <X size={14} aria-hidden="true" />
        </button>
      </div>
      {usage?.available ? (
        <>
          <div className="floating-monitor-line">
            <span>CPU</span>
            <div className="usage-track">
              <i style={{ width: `${Math.min(100, cpu)}%` }} data-level={level(cpu)} />
            </div>
            <b>{cpu.toFixed(0)}%</b>
          </div>
          <div className="floating-monitor-line">
            <span>RAM</span>
            <div className="usage-track">
              <i style={{ width: `${Math.min(100, memory)}%` }} data-level={level(memory)} />
            </div>
            <b>{memory.toFixed(0)}%</b>
          </div>
          {sparkPath ? (
            <svg className="floating-monitor-spark" viewBox="0 0 100 30" preserveAspectRatio="none" aria-label="Histórico de CPU">
              <path d={sparkPath} />
            </svg>
          ) : null}
          <div className="floating-monitor-foot">
            <span>backend: {(usage.process_cpu_percent ?? 0).toFixed(0)}% · {formatGiB(usage.process_memory_bytes)}</span>
            {usage.cpu_per_core ? <span>{usage.cpu_per_core.filter((core) => core > 60).length} núcleo(s) quente(s)</span> : null}
          </div>
        </>
      ) : (
        <div className="floating-monitor-foot">{usage?.reason ?? "carregando…"}</div>
      )}
    </aside>
  );
}
