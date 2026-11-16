import { Braces, FileDown, FileSpreadsheet, FileUp } from "lucide-react";
import { useRef } from "react";
import { InfoTip } from "./InfoTip";

type JsonEditorProps = {
  value: string;
  error: string | null;
  busy?: boolean;
  onChange: (value: string) => void;
  onImportFile: (file: File) => void;
  onDownloadTemplate: () => void;
  onExportXlsx: () => void;
};

export function JsonEditor({
  value,
  error,
  busy = false,
  onChange,
  onImportFile,
  onDownloadTemplate,
  onExportXlsx
}: JsonEditorProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onImportFile(file);
    }
    event.target.value = "";
  };

  return (
    <section className="panel editor-panel" aria-label="Instância JSON">
      <div className="editor-toolbar">
        <div className="panel-heading">
          <Braces size={18} aria-hidden="true" />
          <h2>Instância</h2>
          <InfoTip text="A instância descreve caixa, itens, rotações permitidas e restrições como distância mínima." />
        </div>
        <div className="toolbar-actions" role="group" aria-label="Planilha">
          <button
            type="button"
            className="ghost-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={busy}
            title="Importar instância de uma planilha .xlsx"
          >
            <FileUp size={15} aria-hidden="true" />
            <span>Importar</span>
            <InfoTip text="Lê uma planilha preenchida e converte para o JSON usado pelo solver." />
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={onExportXlsx}
            disabled={busy}
            title="Exportar a instância atual como planilha .xlsx"
          >
            <FileDown size={15} aria-hidden="true" />
            <span>Exportar</span>
            <InfoTip text="Baixa a instância atual como planilha editável." />
          </button>
          <button
            type="button"
            className="ghost-button"
            onClick={onDownloadTemplate}
            disabled={busy}
            title="Baixar planilha modelo com instruções de preenchimento"
          >
            <FileSpreadsheet size={15} aria-hidden="true" />
            <span>Modelo</span>
            <InfoTip text="Baixa um mockup com abas e colunas prontas para preencher." />
          </button>
        </div>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept=".xlsx,.xlsm,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        onChange={handleFileChange}
        hidden
      />
      <textarea
        spellCheck={false}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-invalid={Boolean(error)}
      />
      {error ? <p className="error-line">{error}</p> : null}
    </section>
  );
}
