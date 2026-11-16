# Simulated Annealing no Projeto

Como o SA funciona, como foi adaptado ao 3D-BPP com representação indireta e quais decisões de projeto foram tomadas. Implementação: `src/adaptive_3dbp/metaheuristics/simulated_annealing/`.

## 1. O Algoritmo Clássico

O SA [1] é uma busca local estocástica inspirada no recozimento metalúrgico, formalizada sobre o critério de Metropolis [2]: movimentos que melhoram são sempre aceitos; movimentos que pioram são aceitos com probabilidade `exp(−Δ/T)`, onde a temperatura `T` decresce ao longo do tempo. Temperatura alta = exploração (escapa de ótimos locais); temperatura baixa = intensificação. Sob esquemas de resfriamento suficientemente lentos há garantias assintóticas de convergência [3] — na prática, usa-se resfriamento geométrico e parada por orçamento.

## 2. Adaptação ao 3D-BPP: Representação Indireta

Buscar coordenadas `(x, y, z)` diretamente gera um espaço dominado por soluções inválidas. O SA aqui opera sobre um **candidato discreto**:

```text
candidato = permutação dos itens + rotação preferida por item
```

e um **decoder** geométrico determinístico (Extreme Points [4]) transforma o candidato em placements válidos — verificando caixa, colisões, rotações permitidas, `min_distance`, peso e apoio (seção 3.7). Essa separação busca/decodificação garante que toda solução avaliada é geometricamente válida e mantém o espaço de busca combinatório (permutações), onde os operadores clássicos funcionam bem. O custo é que o mapa candidato→solução é muitos-para-um e não-suave — uma motivação adicional para o SA, robusto a paisagens irregulares.

## 3. Componentes e Decisões

### 3.1 Operadores de vizinhança (`operators.py`)

| Operador | Movimento | Papel |
|---|---|---|
| `swap` | troca dois itens na ordem | perturbação local |
| `relocate` | move um item para outra posição | mudança de prioridade |
| `reverse` | inverte uma subsequência | perturbação média (estilo 2-opt) |
| `rotate` | troca a rotação preferida de um item | dimensão geométrica |
| `ruin_recreate` | remove k itens e reinsere aleatoriamente | diversificação grande |

As probabilidades dos operadores são **parâmetros do espaço de configuração** (simplex), ajustáveis pelo EGO/SMAC/irace — a mistura ótima depende da instância, o que conecta o SA à pesquisa de configuração contextual (`docs/ego.md`). O `ruin_recreate` segue o princípio de *ruin-and-recreate* [5, 6], reconhecido como o mecanismo de diversificação mais eficaz em problemas de empacotamento/roteamento.

### 3.2 Temperatura inicial calibrada por amostragem (`schedule.py`)

Em vez de um valor fixo (frágil, pois a escala de Δ varia com a instância), a temperatura inicial é calibrada num passeio aleatório curto de transições de um passo, resolvendo

```text
T₀ = −Δ̄ / ln(p_aceite_alvo)
```

para uma taxa de aceitação inicial alvo (`target_initial_acceptance`, padrão 0.82). É o método clássico de calibração por taxa de aceite [7]. O alvo de aceite é, ele próprio, um parâmetro configurável pelo tuning. Detalhes da versão 0.2.0:

- **Só deltas dentro do mesmo nível de caixas.** Entram na média apenas os deltas positivos de transições que **mantêm** o número de caixas. Antes, um delta que abria uma caixa (~1000·V na energia antiga) se misturava aos deltas dentro da mesma contagem (≤ V), e T₀ saltava entre uma busca quase gulosa e um passeio quase aleatório conforme a seed. Agora abrir uma caixa é aceito com probabilidade ≈ `exp(−1/T₀)`; nas instâncias de teste da 632591ae, T₀ ≈ 0,015 e essa probabilidade ≈ e⁻⁶⁶.
- **Amostras aproveitáveis.** `temperature_calibration_samples` (padrão 24) é o número de deltas **aproveitáveis** a coletar. Só 10–30% dos passos servem, então o passeio dura até 4× esse número de decodificações. Cada passo conta no orçamento de avaliações.
- **Fallback.** Sem nenhum delta aproveitável (ou com calibração desligada), `T₀ = 0,05 / caixas do start`, porque os deltas dentro do nível escalam como 1/caixas.
- Um start que já está no lower bound não calibra.

Há uma única implementação (`calibrate_initial_temperature`), usada pelo SA e pelo parallel tempering; antes havia duas cópias divergentes.

### 3.3 Resfriamento, estagnação, reaquecimento e parada

Resfriamento geométrico (`T ← α·T`, `cooling_rate`) com `iterations_per_temperature` por patamar — o esquema padrão da prática [1, 3]. Estagnação prolongada (`stagnation_limit` aceites sem melhora) dispara **reaquecimento** (`T ← reheating_factor·T`, até 2 vezes) antes de encerrar o start — uma forma de restart suave que devolve capacidade de escapar de ótimos locais [8].

A temperatura de parada é **relativa** ao T₀ calibrado: `T_parada = max(min_temperature, min_temperature_ratio · T₀)`, com `min_temperature_ratio = 1e-3` e `min_temperature = 1e-9` (só um piso de segurança). Como T₀ cai com o número de caixas, um piso absoluto (o antigo `0,001`) fazia instâncias grandes pararem logo depois da calibração (no MPV c1-n200, com 51 caixas, T₀ chegou a 0,0009). Com o piso relativo toda instância tem o mesmo número de níveis de resfriamento.

### 3.4 Energia escalar (aceitação) e score lexicográfico (incumbente)

Uma ordem lexicográfica não fornece o escalar que o critério de Metropolis exige (`Δ = E(s′) − E(s)`). A **energia usada na aceitação** (versão 0.2.0, `domain/scoring.py`) é adimensional:

```text
E(s) = b(s) + r(s),   r(s) = 1 − média_caixas(u²)  ∈ [0, 1)
```

onde `b` = número de caixas e `u` = ocupação de cada caixa (volume ocupado / volume da caixa). A dominância é imediata: `b₁ < b₂ ⇒ E₁ < b₁ + 1 ≤ b₂ ≤ E₂`. A parte inteira é o número de caixas e a parte fracionária é o gradiente dentro de um mesmo número de caixas; a temperatura passa a ser medida "em caixas" e não depende do volume da caixa.

Com `b` fixo, `média(u)` é constante, e minimizar `r` é maximizar `média(u²)`: a energia prefere concentrar o espaço livre numa caixa (caixas cheias ao lado de uma mais vazia), a direção que ajuda a esvaziar uma caixa e eliminá-la, como na fitness de Falkenauer [15]. Exemplo com duas caixas: `u = [1,0; 0,5]` dá `r = 0,375` e `u = [0,75; 0,75]` dá `r = 0,4375`, e o primeiro é preferido.

O termo `r` é chamado de "fragmentação" por brevidade, mas mede o **desequilíbrio de ocupação entre caixas**, não a geometria dos vazios. O volume vazio total não entra: com todos os itens empacotados ele vale `b·V − Σvᵢ` e é determinado por `b`.

A ordem lexicográfica `(b, r, −c)`, com `c` = compactação média (ocupado / bounding box por caixa), fica restrita à atualização do melhor incumbente, desempate e relatório de qualidade. A compactação não entra na energia.

> **Formas anteriores (até 0.1.x).** A energia era `b·M + r·V`, com `M = max(1000, n+2)·V` e `r = média((1−u)²)`; numa versão ainda anterior havia também o volume vazio total. Com `b` fixo, `média((1−u)²) = 1 − 2·média(u) + média(u²)`, então minimizar esse `r` **igualava** a ocupação das caixas: o oposto do que ajuda a eliminar uma caixa (no exemplo acima, `[0,75; 0,75]` dava 0,0625 e era preferido a `[1,0; 0,5]`, que dava 0,125). A escala `M ≈ 1000·V` também alimentava a bimodalidade da calibração de T₀ (seção 3.2). Histórico em `docs/decisoes_cientificas.md`, seção 8.

### 3.5 Parada por lower bound

Quando o melhor atinge o **lower bound forte** (`docs/decisoes_cientificas.md`, seção 1), a busca para com otimalidade provada (`optimal_proven=true`). O bound é pré-computado por start (não no loop quente) e cacheado por conteúdo geométrico e por identidade da instância.

### 3.6 Multi-start e paralelismo

Vários starts com construtores diferentes (volume decrescente, maior dimensão, maior face, aleatórios com seed) exploram bacias distintas. O pool de starts é cortado em `multi_start_count` (antes, as 3 ordens por regra entravam sempre, e `multi_start_count=1` rodava até 3 starts). No modo sequencial cada start recebe `restante // starts_restantes` avaliações e o que ele não usar (estagnação, lower bound) passa para os seguintes; antes, o primeiro start recebia todo o orçamento restante e `multi_start_count` quase não tinha efeito.

Os modos de execução: `sequential`, `process_pool` (starts independentes, reprodutível) e `cooperative_process_pool` (memória central — ver `docs/paralelismo_cooperativo.md`). O core C++ opcional (protocolo `v2`) recebe o pool inicial do Python, tem decoder bit a bit idêntico ao Python e um SA com a mesma semântica (calibração, reaquecimento, parada no lower bound, multi-start); as trajetórias diferem porque os geradores aleatórios diferem. O desempenho relativo precisa ser medido novamente com a 0.3.0; medições históricas não certificam o speedup atual.

### 3.7 Restrições aplicadas pelo decoder

O decoder Extreme Points (`decoders/extreme_points.py`) e o validador final (`domain/validation.py`) aplicam:

- **rotações permitidas** por item (`all_orthogonal`, `fixed` ou lista); o validador compara dimensões, para não punir itens simétricos;
- **`min_distance`** entre pares de itens na mesma caixa;
- **peso**: uma caixa não recebe um item que faria a soma de `weight` passar de `max_weight`;
- **apoio**: com `support_ratio_min` (ou `stability_required`, que sem razão explícita exige apoio total), a fração da base do item apoiada no piso ou em topos de itens na mesma altura deve atingir o mínimo. O eixo z é tratado como vertical.

Até a 0.1.x, peso, apoio e `stability_required` eram aceitos na instância mas ignorados, e o core C++ podia devolver rotações proibidas como "feasible".

### 3.8 Variante: parallel tempering (`metaheuristics/parallel_tempering/`)

Com `algorithm="parallel_tempering"`, `pt_replicas` cadeias de Metropolis (padrão 8) rodam em temperaturas fixas numa escada geométrica de `T_max` até `T_max · pt_min_temperature_ratio` e, a cada `pt_swap_interval` passos, réplicas vizinhas tentam trocar configurações (pares pares/ímpares alternados). Mesma vizinhança, decoder, energia e contrato do SA, e o mesmo orçamento de avaliações dividido entre as réplicas. Por padrão `T_max` é o T₀ calibrado, e nenhuma réplica cruza níveis de caixas; `pt_max_temperature` (em "caixas"; a escada usa o maior entre ele e T₀) permite uma réplica quente que abre caixas (0,3 dá p ≈ 3,6% de abrir uma caixa). Resfriamento, reaquecimento e estagnação não se aplicam; `pt_replicas` substitui `multi_start_count`. Só no core Python. Referência: Earl e Deem (2005) [13].

## 4. Posição na Literatura e Frentes Recentes

SA + heurística construtiva continua sendo uma linha competitiva e interpretável para 3D-BPP offline, com a vantagem de aceitar restrições novas sem re-treinamento. As frentes recentes mais ativas combinam aprendizado profundo por reforço com heurísticas geométricas — ainda majoritariamente no 3D-BPP *online*: DRL com extreme points/EMS como espaço de ação, transformers para generalização (GOPT, 2024) e híbridos heurística+PPO (HHPPO, 2024) [9, 10, 11]. Para o cenário offline deste projeto, a primeira aposta foi manter o solver clássico e **aprender a configurá-lo** por instância (GPR/EGO — `docs/ego.md`), com validação contra SMAC3/irace. A campanha 632591ae não mostrou ganho significativo, mas rodou com código anterior ao atual e precisa ser refeita. A linha de pesquisa atual (`docs/status.md`) pergunta se aprender **componentes** do SA, como a distribuição de proposta no estilo Neural SA [14], rende mais do que configurar parâmetros; a novidade dessa direção ainda precisa de revisão bibliográfica atualizada e validação experimental.

## Referências

1. Kirkpatrick, S.; Gelatt, C. D.; Vecchi, M. P. *Optimization by Simulated Annealing.* Science, 220(4598), 1983.
2. Metropolis, N.; et al. *Equation of State Calculations by Fast Computing Machines.* J. Chem. Phys., 21, 1953.
3. Aarts, E.; Korst, J. *Simulated Annealing and Boltzmann Machines.* Wiley, 1989.
4. Crainic, T. G.; Perboli, G.; Tadei, R. *Extreme Point-Based Heuristics for Three-Dimensional Bin Packing.* INFORMS Journal on Computing, 20(3), 2008.
5. Schrimpf, G.; et al. *Record Breaking Optimization Results Using the Ruin and Recreate Principle.* Journal of Computational Physics, 159, 2000.
6. Ropke, S.; Pisinger, D. *An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows.* Transportation Science, 40(4), 2006.
7. Ben-Ameur, W. *Computing the Initial Temperature of Simulated Annealing.* Computational Optimization and Applications, 29, 2004.
8. Connolly, D. T. *An Improved Annealing Scheme for the QAP.* European Journal of Operational Research, 46, 1990.
9. Zhao, H.; et al. *Online 3D Bin Packing with Constrained Deep Reinforcement Learning.* AAAI, 2021.
10. Xiong, H.; et al. *GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning.* IEEE RA-L, 2024.
11. *Integrating Heuristic Methods with Deep Reinforcement Learning for Online 3D Bin-Packing Optimization (HHPPO).* Sensors, 24(16), 2024.
12. Martello, S.; Pisinger, D.; Vigo, D. *The Three-Dimensional Bin Packing Problem.* Operations Research, 48(2), 2000.
13. Earl, D. J.; Deem, M. W. *Parallel tempering: theory, applications, and new perspectives.* Physical Chemistry Chemical Physics, 7, 2005. arXiv:physics/0508111.
14. Correia, A. H. C.; Worrall, D. E.; Bondesan, R. *Neural Simulated Annealing.* AISTATS, 2023.
15. Falkenauer, E. *A Hybrid Grouping Genetic Algorithm for Bin Packing.* Journal of Heuristics, 2(1), 1996.
