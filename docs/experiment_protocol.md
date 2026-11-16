# Protocolo Experimental

Este documento orienta campanhas para comparar qualidade, tempo, paralelismo e aprendizado do solver. Registros históricos e manuscritos foram preservados em arquivo privado e não integram a distribuição pública.

> **Estado (14/11/2026).** O código está na versão **0.3.0**, que corrigiu o decoder com first fit e projeção de pontos extremos. As campanhas anteriores, incluindo o piloto MPV 0.2.0 e a `campaign-632591ae`, **não são comparáveis** com o código atual. A próxima campanha confirmatória ainda não foi rodada. Veja [a análise atual](status.md).

## Perguntas De Pesquisa

Numeração canônica (os outros documentos citam estas RQs por número):

- RQ1: O Simulated Annealing com Extreme Points encontra soluções competitivas para instâncias offline 3D-BPP?
- RQ2: O core C++ reduz tempo de parede sem alterar a validade geométrica das soluções?
- RQ3: O multi-start paralelo melhora robustez e tempo até alvo em relação ao sequencial?
- RQ4: O GPR online aprende uma relação útil entre features, configuração e desempenho?
- RQ5: EGO reduz avaliações de configuração em comparação com Random Search?
- RQ6: A configuração aprendida generaliza para instâncias fora da distribuição de treino?
- RQ7: A política de configuração por instância (GP + EGO contextual, *zero-shot*) supera a melhor configuração global com o mesmo orçamento? É a pergunta do primeiro artigo; o braço EGO clássico (sem contexto) e os baselines `per_class` e `nearest_neighbor` medem se o contexto importa.
- RQ8: Se configurar os parâmetros do SA rende pouco, aprender **componentes** do SA (por exemplo, o operador de vizinhança) rende mais, com decoder, orçamento e protocolo fixos? É a pergunta central da linha de pesquisa atual (`docs/status.md`); as comparações previstas incluem SA padrão, SA configurado, parallel tempering e heurísticas clássicas.

RQ1 para o benchmark MPV se responde comparando somas de caixas por (classe, n) com `docs/benchmark_mpv.md`. RQ7 é respondida pela campanha (`run-campaign`). RQ8 ainda não tem braço implementado (Etapa 1 da linha de pesquisa).

## Variáveis Controladas

- Conjunto de instâncias (sintético estratificado ou benchmark MPV) e modo de rotação.
- Seed de construção e SA.
- Algoritmo: `simulated_annealing` ou `parallel_tempering`.
- Core selecionado: Python ou C++ (numa campanha, `budget.native_core` vale para todos os braços).
- Orçamento: avaliações (o tempo é teto de segurança).
- `temperature_calibration_samples` (número de deltas aproveitáveis na calibração de T0).
- Número de starts e número de workers.
- Probabilidades dos operadores.
- Uso ou não de `min_distance`, peso (`max_weight`) e apoio (`stability_required`, `support_ratio_min`).
- Taxa de amostragem da convergência.
- Versão do Python (ver "Reprodutibilidade").

## Conjuntos De Instâncias

**Sintético estratificado** (`generate-dataset`, `adaptive_3dbp/experiments/instances.py`). Classes = tamanho × ocupação × forma × heterogeneidade × `min_distance` (72 classes, ou 36 sem `min_distance`). Regras desde a 0.2.0:

- Só o treino é curado (`curated_splits=("train",)`). Validação e teste são gerados sem rodar o solver.
- A curadoria usa seeds próprias, disjuntas das seeds experimentais, e o tempo não aceita, rejeita nem ordena candidatas: a mesma spec com a mesma seed gera o mesmo dataset.
- **Viés declarado.** A curadoria ainda escolhe *classes* pelo comportamento do painel default no treino (troca classes "fáceis demais"). Não há vazamento de held-out, mas o teste fica condicionado a classes onde o default não resolve tudo de imediato. É um viés de distribuição contra `fixed_default` e deve constar no relato.

**Benchmark de Martello, Pisinger e Vigo (MPV, 2000)** (`adaptive_3dbp/benchmarks/mpv.py`, `dataset.benchmark` na spec; exemplo em `configs/campaigns/mpv_pilot.json`). Porte bit-exato do gerador `test3dbpp.c` (320/320 instâncias idênticas ao OR-Datasets). Regras:

- Orientação fixa por padrão (`rotations: "fixed"`), como nas tabelas publicadas.
- Split por índice v: **teste = as 10 instâncias canônicas (v = 1–10)**, treino v = 11–20, validação v = 21–25. A seed do gerador é v + n e não depende da classe; o gerador recusa splits que compartilhem seeds.
- Sem curadoria: nenhum run do solver escolhe instâncias.
- O nosso lower bound é mais fraco que o L2 de MPV, então o *gap* não se compara com a literatura: compare **somas de caixas por (classe, n)**.

Detalhes: `docs/benchmark_mpv.md` e `docs/benchmark_mpv.md`.

## Métricas De Qualidade

- Caixas usadas.
- Gap para lower bound (métrica primária).
- Gap para best known, quando existir.
- Utilização média das caixas.
- Fragmentação `r = 1 − média(u²)` (desequilíbrio de ocupação entre caixas; `docs/simulated_annealing.md`, seção 3.4).
- Compactação.
- Status final: `feasible`, `infeasible`, `timeout` ou `error`.
- `stop_reason` de cada run: `evaluations`, `lower_bound`, `time`, `converged` ou `failure`.

## Perda De Tuning

A perda que os configuradores minimizam (`performance_loss`, `learning/online_gpr.py`) é

```text
perda = gap_to_lower_bound + 10 · [status ≠ feasible]
```

O tempo de execução **não entra** por padrão: é reportado à parte e só soma à perda com `runtime_weight > 0` explícito. Na `campaign-632591ae`, 6 das 7 "vitórias" do EGO contextual contra o default vinham só do antigo termo `0,01·log1p(runtime)`, medido com 11 runs simultâneos. Não há penalidade por avaliações (com orçamento fixo de avaliações seria dupla contagem).

## Orçamento

- O orçamento de cada run é o número de **avaliações** (decodificações). As amostras de calibração de T0 também consomem avaliações.
- O tempo (`time_limit_seconds`) é teto de segurança. O resumo da campanha conta os `stop_reason` e avisa quando algum run parou por tempo. Na `campaign-632591ae`, 507 de 10.800 runs de treino pararam no limite de 36 s, 442 deles antes das 600 avaliações.
- No multi-start sequencial cada start recebe `restante // starts_restantes`; o que um start não usar passa para os seguintes.
- No parallel tempering as réplicas dividem o mesmo orçamento (decodificações iniciais e calibração incluídas).
- Todos os braços de uma campanha usam o mesmo orçamento em runs do solver. O irace não tem mais o piso de 96 runs; se o próprio irace recusar um orçamento pequeno, o braço aparece como pulado, com o motivo, no relatório.

## Métricas Anytime

Usar `result.convergence` para analisar:

- best-so-far por avaliação.
- bins por avaliação.
- objetivo corrente por avaliação.
- taxa de aceite acumulada.
- número de melhorias.
- tempo até atingir um alvo.
- área sob curva de gap.

## Métricas De Sistemas

Para paralelismo:

- `T1`: tempo sequencial.
- `Tp`: tempo com `p` workers.
- speedup: `T1 / Tp`.
- eficiência: `speedup / p`.
- overhead: tempo de serialização, criação de processo e agregação.
- uso de CPU/RAM.

Para C++:

- tempo médio por avaliação.
- custo de chamada subprocesso.
- diferença de qualidade em relação ao Python (o decoder é bit a bit idêntico; as trajetórias do SA diferem porque os geradores aleatórios diferem).
- validação de restrições no backend.

## Métricas De Aprendizado

Para o GPR:

- quantidade de observações.
- distribuição do target.
- erro em validação separada.
- calibração da incerteza.
- impacto de novas observações no modelo.
- estabilidade por seed.

Para EGO:

- melhor configuração encontrada por rodada.
- Expected Improvement observado.
- comparação contra Random Search.
- custo do configurador em relação ao custo de resolver instâncias.

## Separação De Dados

Usar ao menos três partições:

- `train`: coleta de observações e ajuste inicial.
- `validation`: seleção de configuração/modelo.
- `test`: avaliação final, usada apenas uma vez.

Instâncias do conjunto `test` não devem alimentar o GPR nem a curadoria antes da avaliação final.

Regra de seeds:

```text
tuning:     seed
validation: seed + 10000
test:       seed + 20000
curadoria:  seeds próprias, disjuntas das acima
SMAC3:      seeds de solver derivadas da seed do SMAC, acima de um piso que não colide com as anteriores
```

A recomendação *zero-shot* do EGO contextual para uma instância de validação/teste é o mínimo da **média posterior** sobre as configurações observadas no treino mais um pool semeado pela própria instância. Não depende do estado do gerador aleatório do tuner nem da ordem em que as instâncias são consultadas.

## Protocolo Estatístico

Implementado em `experiments/stats.py`; resultados em `summary.json` (`statistical_comparisons`) e no `report.md`.

- **Unidade.** No sintético, a instância (seeds agregadas por média). No MPV, o par **(n, v)**: classes com o mesmo (n, v) compartilham a seed do gerador (classes 1–5 dividem cerca de 50% dos itens; a classe 6 é função da 8) e viram uma única observação (média). O relatório diz qual unidade foi usada.
- **Teste.** Wilcoxon *signed-rank* pareado: distribuição exata para n ≤ 12; acima, aproximação normal com correção de empates.
- **Efeito.** Correlação rank-biserial pareada; IC 95% *bootstrap* percentil da diferença média (10.000 reamostragens por unidade, seed 0, sem ajuste de multiplicidade); **taxa de vitória** (fração de unidades vencidas, medida descritiva, não uma probabilidade) e contagem vitórias/empates/derrotas (V/E/D).
- **Multiplicidade.** Holm dentro de cada família. Família primária: proposta (`noisy_contextual_ego`) contra `fixed_default`, `random_search`, `classic_ego`, `smac3`, `smac3_ac`, `irace`, `per_class` e `nearest_neighbor` (8 comparações quando todos os braços estão presentes). Família de ablação: `ablation_min_y_incumbent` e `ablation_gp_fixed_hyperparams`.
- **Pendente antes da campanha confirmatória.** Passar de 5 para 8 comparações primárias custa poder. É preciso decidir quais comparações são primárias e dimensionar o número de unidades de teste por análise de poder. Em pilotos, os *p*-valores são descritivos.

## Estimativa De Runs

Para cada braço que faz tuning (todos exceto `fixed_default`):

```text
tuning_runs = trials * train_instances * seeds
eval_runs   = (validation_instances + test_instances) * seeds
total       = tuning_runs + eval_runs
```

Para `fixed_default`: `total = eval_runs`. Ablações e os baselines por instância (`per_class`, `nearest_neighbor`, com o orçamento do Random Search) contam como braços que fazem tuning. A interface mostra a estimativa antes de iniciar a campanha; ela não conta os runs de tuning do irace, que rodam fora do objetivo Python, e zera os braços SMAC3/irace quando a dependência não está instalada.

## Sanity Check E Piloto

**Sanity check local** (verifica pipeline, UI e exports; não é evidência científica):

```text
tuners: fixed_default, random_search, classic_ego, noisy_contextual_ego
classes: 1
train/class: 1
validation/class: 1
test/class: 1
trials: 2
seeds: 11
max_evaluations/run: 80
```

Pela fórmula acima: 3 × 2 + 4 × 2 = 14 runs do solver.

**Pilotos** (só depois do sanity check):

- Sintético: `configs/campaigns/pilot.json` (3 classes, 12 trials, seeds 11/23, 1200 avaliações por run).
- MPV: `configs/campaigns/mpv_pilot.json` (classes 1 e 4–8, n = 50, 7 braços + baselines por instância + ablações, 3000 avaliações por run, `native_core: true`).

Leia a validação para escolher configurações; o teste é lido uma única vez.

## Procedimento Recomendado

1. Commitar o código e conferir que a árvore git está limpa.
2. Validar todas as instâncias com `/api/validate` (ou `validate-instance`).
3. Rodar baseline Python sequencial com múltiplas seeds.
4. Rodar C++ sequencial com o mesmo orçamento.
5. Rodar multi-start paralelo com diferentes números de workers.
6. Rodar a campanha (`run-campaign --spec ...`), que persiste `runs.csv`, `solutions.jsonl`, `metadata.json`, `summary.json` e `report.md`.
7. Consultar `learning_observations` quando a campanha persistir no SQLite.
8. Comparar Random Search, EGO e os demais braços em `validation`.
9. Congelar configuração final.
10. Avaliar uma única vez em `test`.

## Relato Obrigatório

- `campaign_id`.
- Revisão git, `git_dirty` e CRC32 do diff (`git_diff_crc32`), capturados antes de a campanha começar.
- Versão do pacote (`solver_version`) e versão do Python.
- Spec completo.
- Manifesto do dataset ou seed de geração; para o MPV, classes, tamanhos, splits por v e modo de rotação.
- Resumo da curadoria (sintético) e o viés de classe declarado.
- Seeds de tuning, validação e teste.
- Orçamento por run e `temperature_calibration_samples`.
- `native_core` (core usado por todos os braços).
- Hardware.
- Número de runs, contagem de `stop_reason`, falhas e braços pulados (com motivo).
- Unidade estatística e famílias de Holm.
- CSV, metadata e relatório gerados.

## Reprodutibilidade

- **Árvore limpa.** Rode campanhas confirmatórias com o código commitado. Os metadados registram `git_dirty` (inclui arquivos novos não rastreados em `src/` e `configs/`) e o CRC32 de `git diff HEAD -- src configs`; o relatório avisa quando a árvore não estava limpa. As campanhas 632591ae e 6f670fb1 registraram o commit `19506b5`, mas rodaram código que só entrou no commit seguinte.
- **Versão do solver.** O GPR online só usa observações da mesma versão `major.minor` do solver (`persistence/sqlite.py`). Runs salvos pela 0.2 não alimentam o modelo da 0.3.
- **Determinismo.** Um run é função de (instância, configuração, seed) nos modos `sequential` e `process_pool`. O número de workers da campanha muda só o tempo (testes `tests/unit/test_tuning.py::test_parallel_runs_reproduce_serial_exactly` e vizinhos). O modo `cooperative_process_pool` é não-determinístico e se avalia por distribuição.
- **Paridade do core C++.** Protocolo `v2`; o Python manda o próprio pool inicial; o decoder C++ é bit a bit idêntico ao Python (coberto por `tests/unit/test_native_parity.py`), compilado com `-ffp-contract=off`. O binário é recompilado quando o SHA-256 de `cpp/sa_core.cpp` deixa de bater com `artifacts/bin/sa_core.sha256` (não por data de modificação). O SA nativo segue a mesma semântica, mas as trajetórias diferem das do Python porque os geradores aleatórios diferem: use um único core por campanha (`budget.native_core`).
- **Versão do Python.** Empates de `local_waste` no rank do decoder dependem de ruído de ponto flutuante. O decoder reproduz a soma compensada do Python 3.12+; Python 3.11 e 3.12+ diferem em 5 de 402 decodificações. Para reproduzir bit a bit, use a mesma versão menor do Python registrada em `metadata.json` (de preferência ≥ 3.12, para manter a paridade com o C++).
- **Testes.** Em 14/11/2026, 224 testes passaram no snapshot público isolado, com Python 3.14.3 e core nativo obrigatório (1 aviso de depreciação de dependência). O build do frontend também passou. A CI em Python 3.12/3.13 ainda aguarda execução no GitHub.
- **Artefatos grandes.** Os `solutions.jsonl` (até 1,8 GB) não são versionados; `scripts/commit_push.sh` aborta com arquivo acima de 95 MB. Guarde-os fora do git junto com `metadata.json`.

## Cuidados

- Relatar seed, configuração e versão do solver.
- Não comparar tempos de C++ e Python sem informar hardware.
- Não misturar instâncias de teste no aprendizado nem na curadoria.
- Usar mais de uma seed para reduzir conclusões baseadas em sorte.
- Verificar validade geométrica por backend, não apenas pelo core que construiu a solução.
- Não comparar resultados de versões diferentes do solver (por exemplo, 0.2 × 0.3).
- No MPV, não comparar o *gap* com a literatura (lower bounds diferentes) e não tratar classes com o mesmo (n, v) como amostras independentes.
- Não declarar otimalidade sem certificado: `optimal_proven` só vale quando as caixas usadas igualam o lower bound.
- Não preencher o artigo com números de campanhas anteriores à 0.3.0 nem com números simulados.
