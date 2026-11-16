# Decisões Científicas e Justificativas

Registro das correções e melhorias adotadas, com justificativa técnica e referências. Cada seção segue o formato: o que mudou, por quê, e em que fontes a decisão se apoia.

> **Versão atual: 0.3.0 (14/11/2026).** As seções seguintes preservam a evolução das decisões, incluindo correções da 0.2.0. Na 0.3.0 o decoder passou a usar first fit e projeção dos pontos extremos. Resultados de campanhas anteriores não são resultados do código atual; veja [o estado científico](status.md).

## 1. Fortalecimento do Lower Bound (`domain/bounds.py`)

**O que mudou.** O bound deixou de ser apenas volumétrico (L0). O bound exportado (`strong_lower_bound`, mantido sob o nome compatível `simple_lower_bound`) é o máximo de quatro bounds válidos:

1. **L0 contínuo** — `ceil(volume_total / volume_caixa)` [4].
2. **L2 de Martello-Toth sobre a relaxação 1D de volumes** — qualquer packing 3D viável induz um packing 1D dos volumes em caixas de capacidade `V`; logo, o bound L2 do 1D-BPP aplicado aos volumes é válido para o 3D-BPP, inclusive com rotações livres [5].
3. **Bound de clique de conflitos (rotation-aware)** — dois paralelepípedos que cabem individualmente na caixa podem coexistir se e somente se existir um par de rotações e um eixo no qual a soma de suas dimensões cabe na caixa (para exatamente dois itens, não-sobreposição implica separação ao longo de algum eixo). Itens de uma clique do grafo de conflitos exigem caixas distintas. É a generalização, para rotação ortogonal livre, do bound de "itens grandes" L1 de Martello-Pisinger-Vigo [4]; bounds para o caso com rotação são discutidos em [6]. A clique máxima é aproximada por heurística gulosa em múltiplas ordens (válido: qualquer clique é um bound, não precisa ser a máxima).
4. **Clique + volume residual** — os `|Q|` itens da clique abrem `|Q|` caixas; os demais itens, na melhor hipótese, preenchem o espaço livre dessas caixas antes de abrir novas: `LB = |Q| + ceil(max(0, V_resto − V_livre_clique)/V)`. **Nota (out/2026):** algebricamente este bound se reduz a `max(|Q|, L0)`, porque o termo extra vale `max(0, ceil(volume_total/V) − |Q|)`. Ele é válido, mas não acrescenta nada além do tamanho da clique e do L0; foi mantido por explicitar a combinação (docstring de `clique_volume_bound`). Uma variante mais forte empacotaria os itens restantes no espaço livre das caixas da clique por dimensão, não por volume.

**Por quê.** O gap medido contra L0 superestima a distância ao ótimo e contamina o alvo de aprendizado do GPR (`performance_loss` usa `gap_to_lower_bound`). Bounds mais fortes reduzem o ruído estrutural do alvo e aumentam a frequência de `optimal_proven` [4, 6, 7].

**Validade.** Todos os bounds permanecem válidos com rotações ortogonais livres — propriedade verificada por construção e coberta por testes (`tests/unit/test_bounds.py`), incluindo o caso em que a rotação *elimina* conflito.

**Custo.** O grafo de conflitos é O(n²·36). Como `score_solution` consulta o bound a cada decodificação, foi adicionado cache por conteúdo geométrico (o bound depende apenas de caixa + dimensões/rotações dos itens) e a chamada repetida dentro do loop quente do SA foi removida. Montar a chave de conteúdo custa O(n log n) por decodificação; desde a 0.2.0 há também um cache por identidade da instância (instâncias são imutáveis).

**Comparação com a literatura.** O bound combinado é mais fraco que o L2 de Martello, Pisinger e Vigo. No benchmark MPV o *gap* ao nosso bound não se compara com os números publicados: compare somas de caixas por (classe, n) (`docs/benchmark_mpv.md`).

## 2. Correção do EI Ruidoso (`configurators/`)

**O que mudou.** O incumbente do Expected Improvement deixou de ser `min(y)` das médias observadas e passou a ser o **mínimo da média posterior do GP nos pontos observados** (incumbente *plug-in*):

- `classic_ego.py`: novo `plugin_incumbent(gp, X_observado)`.
- `noisy_contextual_ego.py`: incumbente **contextual** — todas as configurações conhecidas são avaliadas pela posterior **sob o contexto da consulta atual**, e o mínimo dessa média posterior é o incumbente.

**Por quê.** Sob ruído, `min(y)` é viesado para baixo ("winner's curse"): uma replicação sortuda faz o alvo parecer melhor que a função latente, encolhe o EI e estagna a busca. O incumbente plug-in pela média posterior é a correção padrão de menor custo [8, 9]; a mesma motivação fundamenta o Noisy EI [10] e correções de forma fechada recentes [11]. No caso contextual, usar `min(y)` também misturaria contextos incomparáveis — instâncias fáceis produziriam incumbentes inalcançáveis para instâncias difíceis.

**Teste.** `test_plugin_incumbent_resists_lucky_outlier` verifica que um outlier sortudo não arrasta o incumbente até `min(y)`.

## 3. Otimização de Hiperparâmetros do GP (`surrogates/exact_gp.py`)

**O que mudou.**

- Lengthscales ARD e variância de sinal agora são estimados por **máxima verossimilhança marginal** (eq. 5.8/5.9 de [12]) com gradiente analítico do kernel Matérn 5/2 e otimizador Adam em log-espaço, com multi-start.
- Regularização por prior log-normal fraco sobre os log-hiperparâmetros (**MAP**), prática padrão para evitar degenerescência com poucos dados [12, 13].
- **Correção de bug**: as variâncias de ruído heterocedástico eram somadas ao kernel na escala original de `y`, enquanto `y` era normalizado — o ruído ficava em escala errada (até ordens de magnitude). Agora `noise/σ_y²`. Coberto por `test_heteroscedastic_noise_is_rescaled_with_target_scale`.
- Gradiente validado contra diferenças finitas (`test_gradient_matches_finite_differences`).

**Por quê.** Lengthscales fixos em 1.0 não fazem ARD de fato: num espaço de ~33 dimensões (contexto + configuração), o surrogate fica não-informativo e o EGO degenera em random search caro. A maximização da verossimilhança marginal é o procedimento canônico [12]; BoTorch/GPyTorch fazem o equivalente (MAP com priors) por padrão [13].

**Custo controlado.** No caminho online, o orçamento de otimização encolhe com o tamanho do dataset (O(n³) por passo); refits de orçamento cheio pertencem a campanhas offline.

## 4. Baselines SMAC3 e irace (`tuning/`)

**O que mudou.** Novo pacote `adaptive_3dbp.tuning` com:

- `objective.py` — objetivo único compartilhado (média de `performance_loss` sobre instâncias × seeds fixos) para todos os tuners; achatamento do simplex de operadores em pesos `op_*` renormalizados (workaround padrão para tuners sem tipo simplex).
- `smac_runner.py` — SMAC3 [14] como dependência opcional (`pip install '.[tuners]'`).
- `irace_runner.py` — geração completa de campanha irace [15] (parameters.txt, scenario.txt, instâncias, target-runner) e execução opcional quando o executável R está no PATH.
- `compare.py` — comparação com orçamento igual: Random Search [16], EGO clássico [2], EGO noise-aware contextual, SMAC3.

**Por quê.** Comparar apenas com random search é insuficiente para publicação: SMAC3 e irace são os baselines esperados em configuração automática de algoritmos [14, 15, 17]. O protocolo (mesmas instâncias, seeds e orçamento; teste usado uma única vez) segue `docs/experiment_protocol.md` e [15].

**Correções de out/2026 (0.2.0).**

- **irace**: mesmo orçamento dos outros braços (o piso de 96 runs saiu); o incumbente é a elite final do próprio irace (antes era a melhor média sobre subconjuntos diferentes de instâncias); o *target runner* usa o interpretador do venv e a mesma calibração e o mesmo core dos outros braços (antes caía em 6 amostras de calibração); quando o irace falha ou recusa o orçamento, o motivo aparece no relatório.
- **SMAC3** (`smac3`): o alvo agora usa a seed que o SMAC atribui ao trial, mapeada para seeds de solver acima de um piso que não colide com as seeds de treino, validação e teste. Antes a seed era ignorada e a intensificação repetia runs idênticos.
- **Novo braço `smac3_ac`**: `AlgorithmConfigurationFacade` com instâncias e features, *racing* entre instâncias e o mesmo orçamento em runs do solver. Contorna um bug do SMAC 2.4 (PCA nas features de instância no treino, mas não na predição marginalizada) com `pca_components=None`. A saída do SMAC vai para a pasta da campanha.
- **Dimensão morta**: `parallel_workers` saiu do espaço de busca (era sempre 1).
- **Core único por campanha**: `budget.native_core` (padrão `false`) vale para todos os braços e para o *target runner* do irace.

## 5. Paralelismo Cooperativo (`metaheuristics/simulated_annealing/cooperative.py`)

**O que mudou.** Novo modo `parallel_mode="cooperative_process_pool"`: workers multi-start trocam a melhor solução global via memória central (`multiprocessing.Manager`), no padrão *central memory / solution warehouse* [18, 19]. A cada `cooperative_sync_interval` avaliações o worker: (a) emite heartbeat; (b) publica seu melhor se superar o global (comparação lexicográfica, sob lock); (c) adota o melhor global se este superar o seu, re-decodificando o candidato (contabilizado como avaliação).

**Sincronismo de relógio: desnecessário.** Em host único todos os processos compartilham a mesma fonte de relógio; e mesmo em cenário distribuído, a ordenação de eventos exigida por este padrão de troca é resolvida por um **sequence number monotônico** (contador `version` atualizado sob lock), não por sincronização física (NTP/Berkeley). Workers decidem "essa informação é nova?" comparando versões, nunca timestamps. A comparação com relógios lógicos de Lamport [20] é apenas analogia conceitual — um contador único sob lock central é estritamente mais simples que o mecanismo de Lamport (que ordena eventos SEM memória central) e não deve ser chamado de equivalente.

**Falhas: detecção por heartbeat; eleição (bully) adiada.** O coordenador é o processo pai — fixo por construção: se ele morre, a computação morre junto, então **eleição de coordenador (algoritmo do valentão [21]) não se justifica em host único**; está documentada como extensão para a versão multi-nó. Worker que trava é detectado por timeout de heartbeat (`heartbeat_timeout_seconds`) e a busca degrada graciosamente (a contribuição dele é descartada; future com exceção vira diagnóstico).

**Python vs C++ para a cooperação.** A coordenação ficou no core Python por decisão deliberada:

1. A troca é de granularidade grossa (a cada N avaliações); pela Lei de Amdahl, a fração sincronizada é desprezível — o ganho de C++ está no caminho quente do decoder, que é ortogonal e pode acelerar cada worker individualmente no futuro.
2. O core C++ se comunica por protocolo textual em subprocessos isolados; memória central em C++ exigiria infra de IPC (shared memory + semáforos ou sockets) com alto custo de implementação/manutenção e duplicação de lógica em duas linguagens (o gap do `min_distance` já mostrou esse risco).
3. As *políticas* de cooperação (o que compartilhar, quando adotar) são o objeto de pesquisa; iterar em Python é mais rápido e seguro.

Caminho híbrido documentado como trabalho futuro: coordenador Python + workers nativos com ponto de sincronização no protocolo textual do core C++.

**Não-determinismo.** O modo cooperativo introduz não-determinismo de escalonamento (a ordem das publicações depende do SO). Para experimentos reprodutíveis seed-a-seed, use `process_pool`; o modo cooperativo deve ser avaliado por distribuição de resultados (médias/quantis sobre repetições), como recomendado para buscas cooperativas [18].

## 6. Robustez do Core Nativo (`native/cpp_bridge.py`)

`OSError`/`SubprocessError` na execução do binário (ex.: binário compilado para outra plataforma — "Exec format error" —, interpretador ausente, timeout) agora viram `NativeCoreError`, acionando o fallback gracioso para o core Python em vez de erro 500 na API.

**Paridade com o Python (out/2026).** O decoder C++ pegava a primeira caixa viável (o Python pega o melhor rank entre todas), ignorava rotações permitidas, peso, apoio e política, e usava outra tolerância; o SA nativo usava a energia antiga e não tinha calibração, reaquecimento, parada no lower bound nem multi-start. Agora o protocolo é `v2` (campos nomeados), o Python manda o próprio pool inicial e o decoder é **bit a bit idêntico** ao Python (732 decodificações em 41 instâncias reais; `tests/unit/test_native_parity.py`). Isso exigiu copiar a soma compensada do Python 3.12+, usar `pow` para `u**2`, arredondamento *half-to-even* e compilar com `-ffp-contract=off`. O SA nativo segue a mesma semântica; as trajetórias diferem porque os geradores aleatórios diferem. O binário é recompilado quando o SHA-256 do fonte muda (antes, por data de modificação). Limitação: empates de `local_waste` no rank dependem de ruído de ponto flutuante, e Python 3.11 e 3.12+ diferem em 5 de 402 decodificações; os metadados da campanha registram a versão do Python.

**Decoder EP-FFD (0.3.0, 21/10/2026).** A paridade acima alinhou o C++ à regra do Python, mas a regra do Python estava errada: o first fit que o C++ usava antes é o do EP-FFD de Crainic et al. (2008). Na 0.3.0 os dois usam first fit entre caixas e projetam os pontos extremos, e a paridade bit a bit continua. Motivo e números: `docs/status.md`.

## 7. Importação de Instâncias por Planilha (`io/spreadsheet.py`)

Suporte a `.xlsx` para facilitar experimentos: template gerado com abas `LEIA-ME`, `meta`, `bin`, `items`, `min_distance`; importação via CLI (`import-spreadsheet`, `spreadsheet-template`, e `solve --instance arquivo.xlsx` direto) e API (`GET /api/examples/spreadsheet-template`, `POST /api/instances/import-spreadsheet`). Round-trip coberto por testes.

## 8. Energia escalar do SA com dominância garantida (`domain/scoring.py`, `cpp/sa_core.cpp`)

**O que é.** A regra de aceitação de Metropolis exige um escalar: `P(aceitar) = exp(−Δ/T)` com `Δ = E(s′) − E(s)`. Uma ordem lexicográfica é uma relação sobre tuplas e não fornece esse escalar automaticamente. A energia usada na aceitação (versão 0.2.0) é adimensional:

```
E(s) = b(s) + r(s),   r(s) = 1 − média_caixas(u²)  ∈ [0, 1)
```

onde `b` = número de caixas e `u` = ocupação de cada caixa (volume ocupado / volume da caixa). Implementada de forma idêntica nos cores Python e C++.

**Condição de dominância (prova).** Como `r ∈ [0, 1)`, `b₁ < b₂ ⟹ E₁ < b₁ + 1 ≤ b₂ ≤ E₂`. Reduzir uma caixa domina qualquer variação do critério secundário, sem nenhuma constante de escala. A parte inteira é o número de caixas; a temperatura passa a ser medida "em caixas" e não depende do volume da caixa.

**Direção do gradiente.** Com `b` fixo e todos os itens empacotados, `média(u) = Σvᵢ / (b·V)` é constante, então minimizar `r` é maximizar `média(u²)`: a energia prefere concentrar o espaço livre numa caixa, a direção que ajuda a esvaziar uma caixa e eliminá-la (fitness de Falkenauer com expoente 2 [30]). Exemplo com duas caixas: `u = [1,0; 0,5]` dá `r = 0,375`; `u = [0,75; 0,75]` dá `r = 0,4375`; o primeiro é preferido. O nome "fragmentação" é operacional: `r` mede o desequilíbrio de ocupação entre caixas, não a geometria dos vazios.

**Papel da ordem lexicográfica.** A ordem usada para incumbente, desempate e relatório de qualidade é `(caixas, fragmentação, −compactação)`. A compactação `c ∈ [0,1]` é a média, sobre caixas, da razão entre volume ocupado e o volume da menor caixa retangular (*bounding box*) que contém os itens daquela caixa; ela não entra na energia. O volume vazio total também fica fora das duas: vale `b·V − Σvᵢ` e é determinado por `b`.

**Histórico e o que foi retirado.**

1. A primeira versão era `f = b·M + e(s) + r·V`, com `e(s)` = volume vazio total. Na resposta ao parecer, `e(s)` foi removido por ser constante dentro de um mesmo `b`, ficando `f = b·M + r·V`, `M = max(1000, n+2)·V`, `r = média((1−u)²)`.
2. A auditoria de 20/10/2026 mostrou dois problemas nessa forma. (a) **Direção errada:** com `b` fixo, `média((1−u)²) = 1 − 2·média(u) + média(u²)`, então minimizar esse `r` **igualava** a ocupação das caixas, o oposto do que ajuda a eliminar uma caixa (no exemplo acima, `[0,75; 0,75]` dava 0,0625 e era preferido a `[1,0; 0,5]`, que dava 0,125). (b) **Escala:** com `M ≈ 1000·V`, um delta que abria uma caixa era ~1000 vezes maior que os deltas dentro do mesmo `b`, o que tornava a calibração de T₀ bimodal (seção 15).
3. A resposta ao parecer afirmava que a remoção de `e(s)` "preservava o comportamento" e que "os resultados da campanha `632591ae` permanecem válidos". **Essa afirmação está retirada.** Ela valia apenas para a remoção de `e(s)`; a campanha rodou com a fragmentação na direção errada e com a calibração bimodal, e seus resultados não são comparáveis com a 0.2.0 (`docs/status.md`).

Teste: `tests/unit/test_constraints_and_energy.py`.

## 9. Validade do bound clique + volume residual (prova e teste)

**Enunciado.** Sejam `Q` uma clique do grafo de conflitos, `V` o volume da caixa, `v_i` o volume do item `i`. Então

```
LB_Q = |Q| + ceil( max(0, Σ_{j∉Q} v_j − R_Q) / V ),   R_Q = |Q|·V − Σ_{i∈Q} v_i
```

é um lower bound válido para o número de caixas.

**Prova.** (i) Dois itens em conflito não podem compartilhar caixa — para exatamente dois paralelepípedos alinhados que cabem individualmente na caixa, coexistir exige separação ao longo de algum eixo em alguma combinação de rotações; a definição de aresta nega exatamente isso. Logo os `|Q|` itens da clique ocupam caixas duas a duas distintas: pelo menos `|Q|` caixas. (ii) A capacidade residual total dessas caixas é no máximo `R_Q = |Q|·V − Σ_{i∈Q} v_i`. O volume dos demais itens que exceder `R_Q` precisa, por relaxação volumétrica, de pelo menos `ceil(excesso/V)` caixas adicionais. ∎

Esta é precisamente a forma com capacidade residual `R_C` exigida em revisão — a soma ingênua `|Q| + ceil(Σ_{j∉Q} v_j / V)` **não** é usada por não ser válida em geral.

**Nota (out/2026).** Como `R_Q = |Q|·V − Σ_{i∈Q} v_i`, o numerador vale `volume_total − |Q|·V`, e `LB_Q = |Q| + max(0, ceil(volume_total/V) − |Q|) = max(|Q|, L0)`. O bound é válido, mas a sua força vem só do tamanho da clique: na tabela de força dos bounds da resposta ao parecer, a linha "clique + volume residual" mede, na prática, a clique (com o L0 como piso).

**Certificação automatizada.** `tests/unit/test_bounds.py::test_bounds_never_exceed_exhaustive_upper_bound` decodifica TODAS as permutações de instâncias pequenas aleatórias (Extreme Points, todas as rotações) e verifica `LB ≤ UB_exaustivo` — qualquer bound acima de uma solução viável seria comprovadamente inválido. `test_clique_volume_bound_accounts_for_residual_capacity` verifica que itens pequenos que cabem no espaço residual da clique não inflam o bound.

## 10. Ruído com shrinkage e piso (`configurators/noisy_contextual_ego.py`, `learning/online_gpr.py`)

**Problema.** Com 2 replicações a variância amostral tem 1 grau de liberdade e pode colapsar para ~0 (GP sobreconfiante, Cholesky instável) ou explodir.

**Solução.** Estimador com shrinkage para a variância global: `σ̃²_i = λ_i·s²_i + (1−λ_i)·σ²_global`, com `λ_i = (r_i−1)/(r_i−1+r₀)` crescendo com o número de replicações `r_i` (`r₀ = 2` pseudo-replicações do prior), mais piso de 5% da variância global. A variância passada ao GP é a do estimador da média (`σ̃²_i / r_i`). No caminho online (runs únicos), aplica-se o piso proporcional no treino.

**EGO clássico (out/2026).** O estimador foi extraído para `configurators/noise.py`. O EGO clássico passou a usar um modelo de ruído análogo: a variância de seed da sua média, com o mesmo piso relativo de 5% (`NOISE_FLOOR_RATIO`), sem o shrinkage. Antes o GP do EGO clássico interpolava sem ruído. Os dois braços diferem agora no contexto e na unidade de observação (média global da configuração × média por instância), não no tratamento do ruído.

## 11. Métrica primária e perda sem tempo de execução (`learning/online_gpr.py`)

A métrica primária do estudo é o **gap ao lower bound** (preserva o objetivo do 3D-BPP: minimizar caixas). Desde a 0.2.0, a perda usada como alvo de tuning é

```
perda = gap + 10·[status ≠ feasible]
```

O tempo de execução é reportado à parte e só entra com `runtime_weight > 0` explícito (`performance_loss(result, runtime_weight=...)`; o padrão `LOSS_WEIGHT_RUNTIME` é 0).

**Por que o termo de tempo saiu.** A versão anterior era `gap + 0.01·log1p(tempo) + 10·falha`. Na `campaign-632591ae`, 7 de 10 instâncias de teste tinham gap idêntico em todos os 7 métodos, e 6 de 7 "vitórias" do EGO contextual contra o default vinham só de `0,01·log1p(runtime)`, medido em parede com 11 runs simultâneos. Um desempate por relógio sob contenção media ruído de agendamento, não qualidade.

A penalidade por avaliações já tinha sido removida antes: com orçamento fixo de avaliações por run, penalizá-las contaria o mesmo recurso duas vezes (e uma avaliação de ruin-and-recreate não custa o mesmo que uma troca simples). Avaliações e tempo seguem reportados como métricas secundárias, e cada run registra `stop_reason` (`evaluations`, `lower_bound`, `time`, `converged`, `failure`).

## 12. Protocolo zero-shot por instância vs. configuração global (`experiments/campaign.py`)

**O problema metodológico.** SMAC, irace, Random Search e EGO clássico resolvem *single best configuration* (`θ* = argmin E_I[c(θ,I)]`); a proposta resolve configuração POR instância (`π(z) = θ*(z)`). Não são o mesmo problema, e a comparação precisa declarar isso.

**O que a campanha faz agora (Protocolo A — zero-shot).**

- Braços globais: tuning no treino → UMA configuração congelada → reavaliada em validação/teste.
- Braço contextual: durante o tuning, cada instância de treino gera observações sob o **seu próprio vetor de features** (a média de contextos apagaria a interação contexto-configuração que a tese investiga); a sugestão de cada trial usa o contexto de uma instância de treino em rodízio. Em validação/teste, a política recomenda **uma configuração por instância usando apenas as features dela** + histórico de treino congelado — nenhuma execução da instância-alvo participa da escolha (zero-shot; sem vazamento por construção). Orçamento idêntico em runs do solver.
- Avaliação **estática** (não prequential): nada do que acontece em validação/teste alimenta tuner algum. O campo `protocol` de cada braço (`global_config` / `per_instance_zero_shot`) registra isso nos artefatos.

**Recomendação *zero-shot* (out/2026).** A política recomendava o argmax do EI num pool aleatório: um critério de exploração, dependente do estado do gerador aleatório (e portanto da ordem em que validação e teste consultavam a política). Agora `NoiseAwareContextualEGO.recommend` devolve o **mínimo da média posterior** sobre as configurações observadas no treino mais um pool semeado pela própria instância. O EI continua sendo usado durante o tuning. O contexto passou a ser as 16 features adimensionais de `FEATURE_NAMES`; antes eram as 19 chaves do extrator, incluindo volumes absolutos. A ablação `ablation_min_y_incumbent` afeta só o EI durante o tuning; a recomendação usa a média posterior nos dois casos.

**Protocolo B (few-shot, k avaliações na instância-alvo) fica como extensão documentada** — exige re-desenho de paridade de informação e não é executado nesta campanha.

## 13. Protocolo estatístico (`experiments/stats.py`)

Unidade experimental: a instância (seeds são replicações internas, agregadas por média antes dos testes). No benchmark MPV a unidade é o par **(n, v)**: as classes com o mesmo (n, v) compartilham a seed do gerador (classes 1–5 dividem cerca de 50% dos itens; a classe 6 é função da 8) e viram uma única observação (média). O relatório diz qual unidade foi usada.

Para cada comparação: Wilcoxon signed-rank pareado (distribuição exata para n ≤ 12; acima, aproximação normal **com correção de empates** — o docstring dizia que havia, o código não tinha até a 0.2.0), correlação **rank-biserial** pareada como tamanho de efeito, IC 95% bootstrap percentil da diferença média, **taxa de vitória** (fração de unidades vencidas; descritiva, não uma probabilidade) e contagem vitórias/empates/derrotas (V/E/D).

**Famílias de Holm.** Holm é aplicado dentro de cada família. Família primária: proposta vs. `fixed_default`, `random_search`, `classic_ego`, `smac3`, `smac3_ac`, `irace`, `per_class` e `nearest_neighbor`. Família de ablação: `ablation_min_y_incumbent` e `ablation_gp_fixed_hyperparams`. A família primária passou de 5 para 8 comparações, o que custa poder; quais comparações são primárias precisa ser decidido antes da campanha confirmatória, com o número de unidades dimensionado por análise de poder (pendente).

Resultados em `summary.json` (`statistical_comparisons`) e seção própria do `report.md`, com aviso explícito de baixo poder em pilotos. Testes em `tests/unit/test_stats.py`.

## 14. Paralelismo de runs da campanha com reprodutibilidade preservada (`tuning/objective.py`)

**O problema.** A campanha executa centenas de runs do solver (braço × instância × seed × trial), mas o objetivo de tuning os rodava estritamente em série — um subprocesso por run, com `process.join()` bloqueante. Numa máquina de 12 núcleos, 11 ficavam ociosos; o piloto de 528 runs levava ~2h14.

**A observação que torna a paralelização segura.** Cada run é uma função pura de `(instância, configuração, seed)`: o SA semeia `random.Random(seed)` e, no modo *Por avaliações*, para pelo orçamento de avaliações (não pelo relógio). Logo o **resultado** de um run independe de quando ou em paralelo com o quê ele roda. Os únicos pontos de estado compartilhado são acumuladores (`records`, `solutions`, contador, persistência) e o callback de progresso — todos no processo pai.

**A implementação (`TuningObjective._run_batch`).** Um pool de threads mantém até *N* execuções de `_solve_with_hard_timeout` em voo. Reusa-se a função já endurecida: cada run continua no seu próprio subprocesso `spawn`, com guarda de timeout/kill individual (não se troca por um `ProcessPoolExecutor`, que perderia o kill por-run). As threads do pai ficam bloqueadas no `join` (liberam o GIL), então o trabalho real corre em paralelo nos núcleos. Callbacks só são disparados pelo pai (nunca por worker), evitando corrida no contador da campanha.

**A garantia de reprodutibilidade.** O *bookkeeping* (records, `runs.csv`, `solutions.jsonl`, persistência SQLite) é montado em **ordem determinística de tarefa** (instância-maior, seed-menor) após a coleta, independente da ordem de conclusão. Assim, o número de workers — e portanto a máquina — muda só o tempo de parede; máquinas com contagens de núcleo diferentes produzem artefatos idênticos. O default é `workers.recommended` (núcleos físicos − 1); o teto é o nº de núcleos físicos (sem sobre-assinatura, preservando a folga do limite de tempo). Sobrescrevível por `spec["max_parallel_runs"]` e registrado em `metadata.json`.

**Tempo de parede e a perda.** Até a 0.1.x a perda incluía um desempate por relógio (`0.01·log1p(runtime_seconds)`), medido em parede e inflado sob contenção. Esta seção dizia que esse ruído era uniforme entre os braços e não enviesava a comparação; a auditoria de 20/10/2026 mostrou que, com 7 de 10 instâncias de teste empatadas no gap, era esse termo que decidia a maior parte das "vitórias" (seção 11). Desde a 0.2.0 o tempo não entra na perda por padrão. A métrica primária (gap ao lower bound) e todas as saídas algorítmicas são bit-a-bit idênticas entre serial e paralelo **enquanto nenhum run para por tempo**; um run que bate no teto de tempo sob contenção pode parar em outro ponto. Na `campaign-632591ae`, 507 de 10.800 runs de treino bateram no limite de 36 s (442 antes das 600 avaliações); por isso o resumo da campanha agora conta os `stop_reason` e avisa quando algum run parou por tempo. O tempo continua dependente de máquina e é reportado à parte. Testes `tests/unit/test_tuning.py::{test_parallel_runs_reproduce_serial_exactly, test_evaluate_configs_matches_serial_loop, test_replicate_losses_grouped_matches_per_instance}` fixam essa invariância. Trials independentes (Random Search) são lotados num único batch (`evaluate_configs`); trials sequenciais (EGO clássico/contextual) paralelizam dentro de cada trial, pois os runs de um trial usam a mesma configuração.

## 15. Calibração de T₀ e temperatura de parada relativa (`metaheuristics/simulated_annealing/schedule.py`, `solver.py`)

**Problema.** T₀ era a média de todos os deltas positivos de 6 amostras. Um delta que abria uma caixa (~1000·V na energia antiga) misturado com deltas dentro da mesma contagem (≤ V) fazia T₀ saltar entre ~0,05·V (busca quase gulosa) e ~5000·V (passeio aleatório: uma caixa a mais aceita com p ≈ 0,8), conforme a seed. É uma explicação plausível para parte do "método = 0,2% da variância" da decomposição feita na resposta ao parecer (não medida separadamente). Havia ainda duas cópias divergentes da calibração.

**Correção.** Passeio aleatório de transições de um passo; só entram deltas positivos que **mantêm** o número de caixas. `T₀ = −média/ln(p₀)`. `temperature_calibration_samples` (padrão 24 na campanha; era 6) é o número de deltas **aproveitáveis** a coletar: só 10–30% dos passos servem, e o passeio dura até 4× esse número de decodificações (com número fixo de passos, T₀ variava 3–4× entre seeds). Sem nenhum delta aproveitável, `T₀ = 0,05 / caixas do start`, porque os deltas dentro do nível escalam como 1/caixas. Medido nas instâncias de teste da 632591ae: deltas médios de 0,002 a 0,004, T₀ ≈ 0,015, e abrir uma caixa tem p ≈ e⁻⁶⁶. Implementação única, usada pelo SA e pelo parallel tempering.

**Parada relativa.** `min_temperature = 0,001` era absoluto, mas o T₀ calibrado cai como 1/caixas: no MPV c1-n200 (51 caixas) T₀ chegou a 0,0009 e o start terminava logo depois da calibração. Agora `T_parada = max(min_temperature, min_temperature_ratio · T₀)`, com `min_temperature_ratio = 1e-3` e `min_temperature = 1e-9` (só piso de segurança): toda instância tem o mesmo número de níveis de resfriamento.

Referência: Ben-Ameur [29] para a calibração por taxa de aceite. Testes em `tests/unit/test_sa.py`.

## 16. Multi-start com orçamento dividido (`metaheuristics/simulated_annealing/solver.py`)

**Problema.** No modo sequencial o primeiro start recebia todo o orçamento restante, e `multi_start_count` quase não tinha efeito. O pool sempre tinha as 3 ordens por regra, então `multi_start_count=1` rodava até 3 starts.

**Correção.** Cada start recebe `restante // starts_restantes`; o que um start não usar (estagnação, lower bound) passa para os seguintes. O pool é cortado em `multi_start_count`. Um start que já está no lower bound não calibra e termina com 1 avaliação (antes gastava até 4× as amostras de calibração).

## 17. Restrições declaradas agora aplicadas (`decoders/extreme_points.py`, `domain/validation.py`)

**Problema.** `max_weight`, `weight`, `stability_required` e `support_ratio_min` eram aceitos no modelo da instância mas ignorados. O core C++ devolvia rotações proibidas como "feasible".

**Correção.** O decoder não usa caixas que estourariam o peso nem posições com apoio abaixo do exigido (`stability_required` sem razão explícita = apoio total; apoio = fração da base sobre o piso ou sobre topos de itens na mesma altura, com z vertical). `validate_solution` rejeita violações de peso e apoio e rotações não permitidas, comparando dimensões para não punir itens simétricos. `fitting_rotation_ids` aceita a mesma folga de 1e-9 do posicionamento e da validação. Teste: `tests/unit/test_constraints_and_energy.py`.

**Alcance.** São restrições geométricas simples (peso total por caixa, apoio de base). Centro de massa, carga sobre itens e verificação física ficam para a Etapa 3 da linha de pesquisa (`docs/status.md`).

## 18. Dataset determinístico e curadoria sem vazamento (`experiments/instances.py`, `experiments/campaign.py`)

**Problema 1 (vazamento).** A curadoria rodava o painel de configuração default nas instâncias de validação e teste, com as seeds de teste, e trocava instâncias "fáceis demais" para o default: o teste ficava enviesado contra o braço `fixed_default`.

**Problema 2 (não determinismo).** A decisão dependia de um *preflight* cronometrado de 2 s: a mesma spec e a mesma seed geraram datasets diferentes nas campanhas 632591ae e 6f670fb1.

**Correção.** Só o treino é curado (`curated_splits=("train",)`); validação e teste são gerados sem rodar o solver. A curadoria usa seeds próprias, disjuntas das seeds experimentais, e o tempo não aceita, rejeita nem ordena candidatas. O painel de curadoria usa a mesma calibração das campanhas (era limitado a 6 amostras). Teste: `test_curation_never_inspects_held_out_splits`.

**Viés que continua valendo (declarar).** A curadoria ainda escolhe *classes* pelo comportamento do painel default no treino (troca classes "fáceis demais"). Não há vazamento de held-out, mas o teste fica condicionado a classes onde o default não resolve tudo de imediato: é um viés de distribuição contra `fixed_default`. O benchmark MPV (seção 20) não tem curadoria.

## 19. Rastreabilidade do código e versão 0.2.0 (`experiments/campaign.py`, `persistence/sqlite.py`)

**Problema.** As campanhas 632591ae e 6f670fb1 registraram o commit `19506b5`, mas rodaram código que só entrou no commit seguinte (a curadoria `prefer_discriminative`), e a energia da campanha não era a Eq. 1 do artigo.

**Correção.** Os metadados registram `git_dirty` (inclui arquivos novos não rastreados em `src/` e `configs/`), o CRC32 de `git diff HEAD -- src configs` e a versão do Python, capturados antes de a campanha começar; o relatório avisa quando a árvore não estava limpa. A versão do pacote passou a **0.2.0** (energia, calibração e perda mudaram). O GPR online só usa observações da mesma versão `major.minor` do solver; runs salvos pela 0.1 aparecem no frontend reescalados para a energia antiga.

## 20. Benchmark de Martello, Pisinger e Vigo (`benchmarks/mpv.py`)

**O que é.** Porte bit-exato do gerador `test3dbpp.c` (versão "new3dbpp"): 320/320 instâncias idênticas às do OR-Datasets, conferidas por SHA-256 em teste. Orientação fixa por padrão, como nas tabelas publicadas. Na campanha entra por `dataset.benchmark` (exemplo: `configs/campaigns/mpv_pilot.json`).

**Split e unidade.** Split por índice v: teste = as 10 instâncias canônicas (v = 1–10), treino v = 11–20, validação v = 21–25. A seed do gerador é v + n e não depende da classe; o gerador recusa splits que compartilhem seeds (a instância (n=50, v=51) são os 50 primeiros itens da canônica (n=100, v=1)). Como as classes com o mesmo (n, v) compartilham itens, a unidade estatística é (n, v) (seção 13).

**Comparação com a literatura.** O nosso lower bound é mais fraco que o L2 de MPV, então o *gap* não se compara: compare somas de caixas por (classe, n) com `docs/benchmark_mpv.md`. Especificação, fontes e armadilhas (versão de 64 bits do RNG, classes 6 e 7 com caixas de 10 e 40) em `docs/benchmark_mpv.md`.

**Por quê.** Na medição de sanidade da fundação, 26 de 30 runs deram o mesmo número de caixas com o código antigo e o novo: as instâncias sintéticas da 632591ae estão perto do teto. O MPV é o benchmark padrão do 3D-BPP offline e tem resultados publicados por (classe, n).

## 21. Parallel tempering (`metaheuristics/parallel_tempering/`)

**O que é.** Réplicas de Metropolis em escada geométrica de temperaturas, com trocas entre vizinhas (esquema par/ímpar alternado) aceitas com `min(1, exp((E_i − E_j)(1/T_i − 1/T_j)))`. Mesmo contrato, vizinhança, decoder, energia e orçamento do SA (`algorithm="parallel_tempering"`). Por padrão `T_max` = T₀ calibrado, e nenhuma réplica cruza níveis de caixas; `pt_max_temperature` (em "caixas") permite uma réplica quente que abre caixas. Só no core Python.

**Por quê.** É um braço de comparação da Etapa 2 da linha de pesquisa: a novidade de parallel tempering neste domínio ainda exige levantamento específico atualizado. Referência: Earl e Deem [31]. Testes em `tests/unit/test_parallel_tempering.py`.

## Referências

1. Kirkpatrick, S.; Gelatt, C. D.; Vecchi, M. P. *Optimization by Simulated Annealing.* Science, 1983.
2. Jones, D. R.; Schonlau, M.; Welch, W. J. *Efficient Global Optimization of Expensive Black-Box Functions.* Journal of Global Optimization, 1998.
3. Crainic, T. G.; Perboli, G.; Tadei, R. *Extreme Point-Based Heuristics for Three-Dimensional Bin Packing.* INFORMS Journal on Computing, 2008.
4. Martello, S.; Pisinger, D.; Vigo, D. *The Three-Dimensional Bin Packing Problem.* Operations Research, 48(2), 2000. (Bounds L0/L1/L2 para 3D-BPP.)
5. Martello, S.; Toth, P. *Knapsack Problems: Algorithms and Computer Implementations.* Wiley, 1990, cap. 8. (Bound L2 do 1D-BPP.)
6. Boschetti, M. A. *New lower bounds for the three-dimensional finite bin packing problem.* Discrete Applied Mathematics, 140(1-3), 2004. (Bounds com e sem rotação.)
7. Fekete, S. P.; Schepers, J. *New classes of fast lower bounds for bin packing problems.* Mathematical Programming, 91, 2001. (Funções duais viáveis; caminho de evolução futura.)
8. Picheny, V.; Wagner, T.; Ginsbourger, D. *A benchmark of kriging-based infill criteria for noisy optimization.* Structural and Multidisciplinary Optimization, 48, 2013. (Comparação de incumbentes/critérios sob ruído; plug-in pela média posterior.)
9. Vazquez, E.; Villemonteix, J.; Sidorkiewicz, M.; Walter, E. *Global optimization based on noisy evaluations: an empirical study of two statistical approaches.* Journal of Physics: Conference Series, 135, 2008.
10. Letham, B.; Karrer, B.; Ottoni, G.; Bakshy, E. *Constrained Bayesian Optimization with Noisy Experiments.* Bayesian Analysis, 14(2), 2019. (Noisy EI.)
11. Zhou, H.; et al. *A Corrected Expected Improvement Acquisition Function Under Noisy Observations.* ACML 2023/PMLR v222, 2024. https://proceedings.mlr.press/v222/zhou24a.html
12. Rasmussen, C. E.; Williams, C. K. I. *Gaussian Processes for Machine Learning.* MIT Press, 2006, cap. 5 (eq. 5.8/5.9: verossimilhança marginal e gradiente).
13. Snoek, J.; Larochelle, H.; Adams, R. P. *Practical Bayesian Optimization of Machine Learning Algorithms.* NeurIPS, 2012. (Tratamento de hiperparâmetros do GP em BO; priors/MAP/marginalização.)
14. Lindauer, M.; Eggensperger, K.; Feurer, M.; et al. *SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization.* JMLR 23, 2022.
15. López-Ibáñez, M.; Dubois-Lacoste, J.; Pérez Cáceres, L.; Birattari, M.; Stützle, T. *The irace package: Iterated racing for automatic algorithm configuration.* Operations Research Perspectives, 3, 2016.
16. Bergstra, J.; Bengio, Y. *Random Search for Hyper-Parameter Optimization.* JMLR 13, 2012.
17. Hutter, F.; Hoos, H. H.; Leyton-Brown, K. *Sequential Model-Based Optimization for General Algorithm Configuration.* LION, 2011.
18. Crainic, T. G. *Parallel Metaheuristics and Cooperative Search.* In: Handbook of Metaheuristics, 3a ed., Springer, 2019. (Central memory / solution warehouse.)
19. Crainic, T. G.; Toulouse, M. *Parallel Meta-heuristics.* In: Handbook of Metaheuristics, 2a ed., Springer, 2010.
20. Lamport, L. *Time, Clocks, and the Ordering of Events in a Distributed System.* Communications of the ACM, 21(7), 1978.
21. Garcia-Molina, H. *Elections in a Distributed Computing System.* IEEE Transactions on Computers, C-31(1), 1982. (Algoritmo do valentão.)
22. Kadioglu, S.; Malitsky, Y.; Sellmann, M.; Tierney, K. *ISAC — Instance-Specific Algorithm Configuration.* ECAI, 2010. (Baseline clássico de configuração por instância.)
23. Schede, E.; et al. *A Survey of Methods for Automated Algorithm Configuration.* JAIR 75, 2022. (Taxonomia: single best vs. per-instance configuration.)
24. Eggensperger, K.; Lindauer, M.; Hutter, F. *Pitfalls and Best Practices in Algorithm Configuration.* JAIR 64, 2019. (Paridade de orçamento, vazamento, protocolos.)
25. Demšar, J. *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7, 2006. (Wilcoxon pareado; testes não-paramétricos.)
26. García, S.; Herrera, F. *An Extension on Statistical Comparisons... for All Pairwise Comparisons.* JMLR 9, 2008. (Holm e procedimentos de multiplicidade.)
27. Adriaensen, S.; et al. *Automated Dynamic Algorithm Configuration.* JAIR 75, 2022. (Configuração dinâmica por fase — roadmap.)
28. Rardin, R. L.; Uzsoy, R. *Experimental Evaluation of Heuristic Optimization Algorithms: A Tutorial.* Journal of Heuristics 7, 2001. (Desenho experimental.)
29. Ben-Ameur, W. *Computing the Initial Temperature of Simulated Annealing.* Computational Optimization and Applications, 29, 2004. (Calibração de T₀ por taxa de aceite.)
30. Falkenauer, E. *A Hybrid Grouping Genetic Algorithm for Bin Packing.* Journal of Heuristics, 2(1), 1996. (Fitness que favorece caixas cheias.)
31. Earl, D. J.; Deem, M. W. *Parallel tempering: theory, applications, and new perspectives.* Physical Chemistry Chemical Physics, 7, 2005. (Replica exchange.)
