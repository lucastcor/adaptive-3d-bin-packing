import { Activity, Cpu, HardDrive, Info, MonitorCog } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { fetchSystemUsage } from "../lib/api";
import type { SystemInfo, SystemUsage } from "../types";
import { InfoTip } from "./InfoTip";

type SystemPanelProps = {
  info: SystemInfo | null;
};

const HISTORY_LENGTH = 60;

function formatBytes(bytes: number | null | undefined) {
  if (!bytes || bytes <= 0) {
    return "-";
  }
  const gib = bytes / 1024 ** 3;
  if (gib >= 1) {
    return `${gib.toLocaleString("pt-BR", { maximumFractionDigits: 1 })} GiB`;
  }
  return `${(bytes / 1024 ** 2).toLocaleString("pt-BR", { maximumFractionDigits: 0 })} MiB`;
}

function Sparkline({ values }: { values: number[] }) {
  const width = 240;
  const height = 48;
  if (values.length < 2) {
    return <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} aria-hidden="true" />;
  }
  const path = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = height - (Math.min(value, 100) / 100) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label="Histórico de CPU">
      <path d={`${path} L ${width} ${height} L 0 ${height} Z`} className="sparkline-fill" />
      <path d={path} className="sparkline-line" />
    </svg>
  );
}

function UsageBar({ label, percent, detail }: { label: string; percent: number; detail?: string }) {
  const clamped = Math.min(100, Math.max(0, percent));
  return (
    <div className="usage-bar">
      <div className="usage-bar-heading">
        <span>{label}</span>
        <strong>{clamped.toLocaleString("pt-BR", { maximumFractionDigits: 1 })}%</strong>
      </div>
      <div className="usage-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={clamped}>
        <i style={{ width: `${clamped}%` }} data-level={clamped > 85 ? "high" : clamped > 60 ? "mid" : "low"} />
      </div>
      {detail ? <small>{detail}</small> : null}
    </div>
  );
}

export function SystemPanel({ info }: SystemPanelProps) {
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
          historyRef.current = [...historyRef.current, next.cpu_percent].slice(-HISTORY_LENGTH);
          setHistory(historyRef.current);
        }
      } catch {
        if (active) {
          setUsage({ available: false, reason: "falha ao consultar /api/system/usage" });
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

  return (
    <div className="system-page">
      <section className="panel system-card" aria-label="Máquina">
        <div className="panel-heading">
          <MonitorCog size={18} aria-hidden="true" />
          <h2>Esta máquina</h2>
          <InfoTip text="O backend detecta SO, arquitetura, núcleos e memória para recomendar quantos workers paralelos fazem sentido aqui." />
        </div>
        {info ? (
          <dl className="system-details">
            <div>
              <dt>Sistema</dt>
              <dd>
                {info.os} {info.os_release}
              </dd>
            </div>
            <div>
              <dt>Arquitetura</dt>
              <dd>{info.architecture}</dd>
            </div>
            <div>
              <dt>Processador</dt>
              <dd>{info.processor || info.architecture}</dd>
            </div>
            <div>
              <dt>Núcleos</dt>
              <dd>
                {info.cpu_physical ?? "?"} físicos / {info.cpu_logical} lógicos
              </dd>
            </div>
            <div>
              <dt>Memória total</dt>
              <dd>{formatBytes(info.total_memory_bytes)}</dd>
            </div>
            <div>
              <dt>GPU</dt>
              <dd>{info.gpus.length > 0 ? info.gpus.join(", ") : "não detectada"}</dd>
            </div>
            <div>
              <dt>Core C++</dt>
              <dd>{info.native_core_available ? "disponível" : "indisponível"}</dd>
            </div>
            <div>
              <dt>Pausa de subprocessos</dt>
              <dd>{info.pause_signals_available ? "suportada (SIGSTOP)" : "indisponível neste SO"}</dd>
            </div>
          </dl>
        ) : (
          <div className="chart-empty">Carregando informações da máquina…</div>
        )}
        <div className="analysis-note">
          <Info size={15} aria-hidden="true" />
          <span>
            O solver é <strong>CPU-bound</strong>: a GPU não é usada na busca. Workers além dos núcleos físicos geram
            contenção em vez de speedup — recomendado aqui: <strong>{info?.workers.recommended ?? "…"}</strong> (máx.{" "}
            {info?.workers.maximum ?? "…"}).
          </span>
        </div>
      </section>

      <section className="panel system-card" aria-label="Consumo em tempo real">
        <div className="panel-heading">
          <Activity size={18} aria-hidden="true" />
          <h2>Consumo em tempo real</h2>
          <InfoTip text="Atualizado a cada 2 segundos. Use durante uma execução para ver o efeito do paralelismo nos núcleos." />
        </div>
        {usage?.available ? (
          <div className="usage-grid">
            <div className="usage-block">
              <UsageBar
                label="CPU total"
                percent={usage.cpu_percent ?? 0}
                detail={usage.load_average ? `load: ${usage.load_average.join(" / ")}` : undefined}
              />
              <Sparkline values={history} />
            </div>
            <div className="usage-block">
              <UsageBar
                label="Memória"
                percent={usage.memory_percent ?? 0}
                detail={`${formatBytes(usage.memory_used_bytes)} de ${formatBytes(usage.memory_total_bytes)}`}
              />
              <div className="usage-bar-heading process-line">
                <span>
                  <Cpu size={14} aria-hidden="true" /> Processo do backend
                </span>
                <strong>
                  {`${(usage.process_cpu_percent ?? 0).toLocaleString("pt-BR", { maximumFractionDigits: 0 })}% · ${formatBytes(
                    usage.process_memory_bytes
                  )}`}
                </strong>
              </div>
            </div>
            <div className="core-grid" aria-label="Uso por núcleo">
              {(usage.cpu_per_core ?? []).map((core, index) => (
                <div className="core-cell" key={index} title={`Núcleo ${index}: ${core.toFixed(0)}%`}>
                  <i
                    style={{ height: `${Math.min(100, Math.max(3, core))}%` }}
                    data-level={core > 85 ? "high" : core > 60 ? "mid" : "low"}
                  />
                  <span>{index}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="chart-empty">
            <span>
              <HardDrive size={16} aria-hidden="true" /> {usage?.reason ?? "Monitor indisponível"}
            </span>
          </div>
        )}
      </section>
    </div>
  );
}
