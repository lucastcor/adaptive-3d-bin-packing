import { BarChart3, BookOpen, Clock, Database, Download, FileText, Settings, X } from "lucide-react";
import { useEffect } from "react";

type CampaignHelpDialogProps = {
  onClose: () => void;
  datasetMode: "generated" | "manifest";
  classesLimit: number;
  perClassTrain: number;
  trainInstances: number;
  validationInstances: number;
  testInstances: number;
  totalGeneratedInstances: number;
  trials: number;
  seedList: string;
  validationSeedList: string;
  testSeedList: string;
  maxEvaluations: number;
  timeLimit: number;
  occupancyScale: number;
  includeMinDistance: boolean;
  persist: boolean;
  selectedArmCount: number;
  estimatedRuns: number | null;
  estimatedDuration: string;
};

function formatInteger(value: number) {
  return new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 }).format(value);
}

export function CampaignHelpDialog({
  onClose,
  datasetMode,
  classesLimit,
  perClassTrain,
  trainInstances,
  validationInstances,
  testInstances,
  totalGeneratedInstances,
  trials,
  seedList,
  validationSeedList,
  testSeedList,
  maxEvaluations,
  timeLimit,
  occupancyScale,
  includeMinDistance,
  persist,
  selectedArmCount,
  estimatedRuns,
  estimatedDuration
}: CampaignHelpDialogProps) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const generatedMode = datasetMode === "generated";
  const runSummary =
    estimatedRuns === null
      ? "o número de runs será calculado depois que o manifest for lido"
      : `${formatInteger(estimatedRuns)} runs estimados, cerca de ${estimatedDuration}`;

  return (
    <div className="campaign-help-overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="campaign-help-dialog" role="dialog" aria-modal="true" aria-labelledby="campaign-help-title">
        <header className="campaign-help-header">
          <div>
            <BookOpen size={20} aria-hidden="true" />
            <div>
              <span>Ajuda da campanha</span>
              <h2 id="campaign-help-title">Como configurar, rodar e entender os resultados</h2>
            </div>
          </div>
          <button className="help-close-button" type="button" onClick={onClose} aria-label="Fechar ajuda">
            <X size={18} aria-hidden="true" />
          </button>
        </header>

        <div className="campaign-help-body">
          <section className="help-lead-card">
            <strong>Resposta curta</strong>
            <p>
              A campanha é um teste controlado: ela cria ou lê várias instâncias, deixa cada estratégia escolher uma
              receita de parâmetros para o mesmo solver e depois compara essas receitas em validação e teste. Ela não
              usa a instância aberta na aba Solver. No modo <strong>Gerado</strong>, o dataset nasce quando você clica
              em Rodar campanha; no modo <strong>Manifest local</strong>, ela usa um dataset já salvo no seu disco.
            </p>
            <div className="help-current-run">
              <span>{selectedArmCount} estratégias selecionadas</span>
              <span>{runSummary}</span>
              <span>{persist ? "SQLite ligado" : "SQLite desligado"}</span>
            </div>
          </section>

          <section className="help-panel">
            <div className="help-panel-heading">
              <Database size={17} aria-hidden="true" />
              <h3>1. O que é o dataset da campanha?</h3>
            </div>
            <div className="help-dataset-layout">
              <div className="dataset-sketch" aria-label="Desenho de uma classe do dataset">
                <div className="dataset-class-box">
                  <span>Classe de instância</span>
                  <strong>tamanho · ocupação · forma · heterogeneidade · distância</strong>
                </div>
                <div className="dataset-splits">
                  <span>treino</span>
                  <b>{generatedMode ? trainInstances : "manifest"}</b>
                  <span>validação</span>
                  <b>{generatedMode ? validationInstances : "manifest"}</b>
                  <span>teste</span>
                  <b>{generatedMode ? testInstances : "manifest"}</b>
                </div>
              </div>
              <div>
                <p>
                  Cada instância usa caixas de <strong>100 × 100 × 100</strong>. O volume total dos itens é calibrado
                  em <strong>múltiplos do volume de uma caixa</strong>: classe <em>low</em> ≈ 1,6 caixas e{" "}
                  <em>high</em> ≈ 3,0 caixas antes do fator de dificuldade. Nesta tela o fator é{" "}
                  <strong>{occupancyScale.toFixed(1)}×</strong>, então a campanha procura instâncias com mais caixas
                  e maior chance de diferenciar os parâmetros do SA. Se tudo coubesse
                  em 1 caixa, todos os configuradores empatariam e a comparação não mediria nada.
                </p>
                <p>
                  No modo gerado, a campanha faz uma curadoria rápida: descarta candidatos lentos demais, evita
                  instâncias triviais e prioriza candidatos em que configurações fixas diferentes produzem diferença de
                  caixas ou gap. Quando uma classe não gera instância boa, ela pode ser substituída por outra classe do
                  mesmo pool.
                </p>
                <p>
                  As classes variam: quantidade de itens (<strong>small</strong> 10-20, <strong>medium</strong> 30-60,{" "}
                  <strong>large</strong> 80-140), forma dos itens, heterogeneidade e opção de <code>min_distance</code>.
                </p>
                <p>
                  {generatedMode
                    ? `Com os valores atuais, serão geradas ${totalGeneratedInstances} instâncias: ${trainInstances} para treino, ${validationInstances} para validação e ${testInstances} para teste.`
                    : "Como você escolheu Manifest local, a tela só saberá as contagens reais depois que o backend ler o manifest.json."}
                </p>
              </div>
            </div>
          </section>

          <section className="help-panel">
            <div className="help-panel-heading">
              <Settings size={17} aria-hidden="true" />
              <h3>2. Para que serve cada parâmetro?</h3>
            </div>
            <div className="help-param-grid">
              <article>
                <strong>Trials</strong>
                <p>
                  Pense em trial como uma tentativa de receita. Em cada trial, a estratégia escolhe uma combinação de
                  parâmetros do SA e testa essa receita nas instâncias de treino. Com 6 trials, cada estratégia tenta 6
                  receitas antes de escolher a melhor para validação.
                </p>
                <span>Atual: {trials}</span>
              </article>
              <article>
                <strong>Seeds</strong>
                <p>
                  Seed é o número que controla o sorteio. A mesma seed repete o mesmo caminho aleatório; seeds
                  diferentes testam se o resultado não foi sorte. Todas as estratégias usam as mesmas seeds para a
                  comparação ser justa.
                </p>
                <span>Treino {seedList} · validação {validationSeedList} · teste {testSeedList}</span>
              </article>
              <article>
                <strong>Aval./run</strong>
                <p>
                  É o número máximo de movimentos que o solver pode tentar em uma run. Uma run é uma rodada pequena:
                  uma instância, uma seed e uma receita de parâmetros.
                </p>
                <span>Atual: {formatInteger(maxEvaluations)} avaliações</span>
              </article>
              <article>
                <strong>Tempo/run</strong>
                <p>
                  É o teto de segundos para uma run individual. Se uma run passar desse tempo, ela entra como timeout
                  e recebe penalidade; a campanha inteira demora mais porque soma muitas runs.
                </p>
                <span>Atual: {timeLimit}s por run</span>
              </article>
              <article>
                <strong>Classes</strong>
                <p>
                  Classe é um tipo de problema sintético, por exemplo pequeno ou grande, baixa ou alta ocupação,
                  itens cúbicos ou alongados. Mais classes significa testar em problemas mais variados.
                </p>
                <span>Atual: {classesLimit}</span>
              </article>
              <article>
                <strong>Train/classe</strong>
                <p>
                  Quantos exemplos de treino cada classe ganha. Se Classes = 4 e Train/classe = 2, o treino terá 8
                  instâncias. Validação e teste recebem 1 por classe nesta tela.
                </p>
                <span>Atual: {perClassTrain}</span>
              </article>
              <article>
                <strong>min_distance</strong>
                <p>
                  Inclui instâncias com restrição de distância mínima entre pares de itens. Isso deixa algumas classes
                  mais difíceis.
                </p>
                <span>{includeMinDistance ? "Ligado" : "Desligado"}</span>
              </article>
              <article>
                <strong>Persistir execuções</strong>
                <p>
                  Salva cada run no banco SQLite com o campaign_id. Ligue quando quiser auditoria e histórico completo;
                  deixe desligado em pilotos rápidos para não misturar testes no aprendizado online.
                </p>
                <span>{persist ? "Ligado" : "Desligado"}</span>
              </article>
            </div>
          </section>

          <section className="help-panel">
            <div className="help-panel-heading">
              <Clock size={17} aria-hidden="true" />
              <h3>3. Por que a campanha pode demorar?</h3>
            </div>
            <div className="help-run-formula">
              <div>
                <span>1 run do solver</span>
                <strong>1 instância × 1 seed × 1 configuração</strong>
              </div>
              <i />
              <div>
                <span>Treino de uma estratégia</span>
                <strong>classes × train/classe × seeds × trials</strong>
              </div>
              <i />
              <div>
                <span>Validação + teste</span>
                <strong>classes × 2 × seeds</strong>
              </div>
            </div>
            <p className="help-text-line">
              A configuração fixa não tem trials porque não escolhe receita nova; ela só roda validação e teste. Random
              Search, EGO, SMAC3, irace e ablations usam trials no treino, depois passam por validação e teste.
            </p>
          </section>

          <section className="help-panel">
            <div className="help-panel-heading">
              <BookOpen size={17} aria-hidden="true" />
              <h3>4. Receitas práticas para escolher valores</h3>
            </div>
            <div className="help-recipe-grid">
              <article>
                <strong>Entender se está funcionando</strong>
                <p>1 classe, train/classe 1, 2 trials, seed 11, 80 aval./run, 1s/run.</p>
              </article>
              <article>
                <strong>Piloto mais confiável</strong>
                <p>4-6 classes, train/classe 2, 8-12 trials, seeds 11 e 23, 600-1200 aval./run, modo por avaliações.</p>
              </article>
              <article>
                <strong>Campanha séria</strong>
                <p>
                  Use manifest curado ou dataset gerado com 6+ classes, 2+ treino/classe, 20+ trials e 3+ seeds.
                  Defina tudo antes, rode uma vez e não use o teste para ficar ajustando parâmetros.
                </p>
              </article>
            </div>
          </section>

          <section className="help-panel">
            <div className="help-panel-heading">
              <BarChart3 size={17} aria-hidden="true" />
              <h3>5. Como analisar os resultados?</h3>
            </div>
            <div className="help-results-flow">
              <div>
                <span>Validação</span>
                <strong>principal</strong>
                <p>Use para escolher a estratégia vencedora. Menor perda é melhor.</p>
              </div>
              <div>
                <span>Teste</span>
                <strong>olhar uma vez</strong>
                <p>Confirma se a escolha generalizou. Depois de olhar, não ajuste a campanha por ele.</p>
              </div>
              <div>
                <span>Caixas e gap</span>
                <strong>qualidade</strong>
                <p>Mostram se a perda veio de usar mais caixas ou ficar longe do limite inferior.</p>
              </div>
              <div>
                <span>Status</span>
                <strong>ok / timeout / skipped</strong>
                <p>Timeout indica run interrompida pelo Tempo/run; skipped indica dependência ausente.</p>
              </div>
              <div>
                <span>Aviso de dataset</span>
                <strong>diagnóstico</strong>
                <p>Se todos batem o lower bound em poucas avaliações, o empate é esperado e não serve como tabela final.</p>
              </div>
            </div>
          </section>

          <section className="help-panel">
            <div className="help-panel-heading">
              <Download size={17} aria-hidden="true" />
              <h3>6. O que vem nos downloads?</h3>
            </div>
            <div className="help-download-grid">
              <article>
                <FileText size={16} aria-hidden="true" />
                <strong>runs.csv</strong>
                <p>Planilha granular: cada execução do solver com instância, seed, estratégia, fase, perda, bins, gap e status.</p>
              </article>
              <article>
                <FileText size={16} aria-hidden="true" />
                <strong>solutions.jsonl</strong>
                <p>Auditoria das soluções: placements por item, métricas de cada bin e curva de convergência de cada run.</p>
              </article>
              <article>
                <FileText size={16} aria-hidden="true" />
                <strong>report.md</strong>
                <p>Relatório legível com tabela final e contexto da campanha.</p>
              </article>
              <article>
                <FileText size={16} aria-hidden="true" />
                <strong>metadata.json</strong>
                <p>Auditoria: versão do código, hardware, seeds, orçamento, splits e configuração completa.</p>
              </article>
              <article>
                <FileText size={16} aria-hidden="true" />
                <strong>manifest dataset</strong>
                <p>Lista dos arquivos JSON das instâncias usadas; serve para repetir a mesma campanha depois.</p>
              </article>
            </div>
          </section>
        </div>
      </section>
    </div>
  );
}
