# EGO — Efficient Global Optimization no Projeto

Como o EGO funciona, como é usado para configurar o Simulated Annealing e quais adaptações foram feitas (incumbente sob ruído, variante contextual, baselines). Implementação: `src/adaptive_3dbp/configurators/`.

## 1. O Que É

EGO [1] é otimização bayesiana para funções caras de avaliar: em vez de testar configurações às cegas, ajusta-se um surrogate GP aos resultados já vistos e escolhe-se a próxima configuração maximizando uma função de aquisição. A aquisição clássica é o **Expected Improvement**:

```text
EI(x) = E[max(0, f_best − f(x))]
      = (f_best − μ(x)) Φ(z) + σ(x) φ(z),   z = (f_best − μ(x)) / σ(x)
```

O EI equilibra automaticamente **explotação** (μ baixo) e **exploração** (σ alto) [1, 2]. Aqui, "função cara" é literal: cada avaliação de configuração significa rodar o solver inteiro em instâncias reais.

## 2. Como o Projeto Usa

O alvo da otimização é a mesma `performance_loss` do aprendizado online (`docs/aprendizado_online.md`), sobre o espaço de busca declarativo em `configs/search_spaces/sa_search_space.json` (taxa de resfriamento, iterações por temperatura, limite de estagnação, tamanho do ruin, multi-start, probabilidades dos operadores etc.).

Fluxo (igual nos dois configuradores):

```text
1. enquanto houver poucas observações: sugestões aleatórias (design inicial)
2. ajustar GP às observações (hiperparâmetros por MLE/MAP — docs/gpr.md)
3. amostrar um pool de candidatos do espaço de busca (512 por padrão)
4. calcular EI de cada candidato
5. devolver os de maior EI; avaliar no solver; registrar; voltar a 2
```

A maximização do EI por pool amostrado é a escolha pragmática para espaço misto (contínuo + inteiro + simplex de operadores); a alternativa com gradiente exigiria relaxações.

O EI é o critério para escolher **o que testar a seguir** durante o tuning. Para **recomendar** uma configuração sem rodar mais nada (o caso *zero-shot* em validação/teste), o EGO contextual usa outro critério desde a 0.2.0: o mínimo da **média posterior** sobre as configurações observadas no treino mais um pool semeado pela própria instância (`NoiseAwareContextualEGO.recommend`). Antes a recomendação era o argmax do EI num pool aleatório, um critério de exploração que dependia do estado do gerador aleatório e da ordem em que as instâncias eram consultadas.

## 3. Adaptações Feitas

### 3.1 Incumbente plug-in sob ruído (correção do "winner's curse")

O EI clássico assume observações sem ruído e usa `f_best = min(y)`. Com avaliações ruidosas (seeds!), `min(y)` é viesado para baixo: uma replicação sortuda faz o alvo parecer melhor do que a função latente, encolhe o EI e estagna a busca. A correção adotada é o **incumbente plug-in**: `f_best = min` da **média posterior do GP** nos pontos observados [3, 4] — mesma motivação do Noisy EI [5] e de correções de forma fechada recentes [6], a uma fração do custo. Implementado em `classic_ego.plugin_incumbent` e coberto por teste (outlier sortudo não arrasta o incumbente).

### 3.2 Variante contextual com variância observacional informada (`noisy_contextual_ego.py`)

**Nome preciso (parecer):** "EI contextual com variância observacional informada e incumbente plug-in pela média posterior". O rótulo "ciente de ruído" deve ser lido nesse sentido — a implementação NÃO integra a incerteza do incumbente (Noisy EI completo [5] e Knowledge Gradient ficam como extensões declaradas). Duas extensões sobre o EGO clássico:

- **Ruído heterocedástico por replicação (com shrinkage):** cada configuração é avaliada em múltiplos seeds; a variância empírica passa por shrinkage para a variância global + piso (com 2 seeds, a variância amostral tem 1 grau de liberdade e colapsa/explode — `docs/decisoes_cientificas.md`, seção 10) e, dividida pelo número de replicações, entra como ruído da observação no GP. Na predição, o EI usa a **variância latente** (sem ruído observacional), evitando preferir regiões ruidosas. Desde a 0.2.0 o EGO clássico também recebe ruído: a variância de seed da sua média, com o mesmo piso relativo de 5% (`NOISE_FLOOR_RATIO` em `configurators/noise.py`), mas sem o shrinkage; antes o GP dele interpolava sem ruído. Os dois braços diferem no contexto e na unidade de observação (média global × média por instância).
- **Contexto da instância:** o vetor do GP é `[features da instância ; configuração]`, com as 16 features adimensionais de `FEATURE_NAMES` (antes da 0.2.0 eram as 19 chaves do extrator, incluindo volumes absolutos). Isso transforma o problema em **configuração por instância** (*per-instance algorithm configuration* [7, 8]): o mesmo modelo generaliza entre instâncias e a sugestão é condicionada às features da instância em mãos — formalmente, otimização bayesiana contextual [9].
- **Incumbente contextual:** o incumbente é o mínimo da média posterior avaliando **todas as configurações conhecidas sob o contexto da consulta atual**. Usar `min(y)` aqui seria duplamente errado: viés de ruído + mistura de contextos incomparáveis (uma instância fácil produziria um incumbente inalcançável para uma difícil).

### 3.3 Baselines para validação honesta (`tuning/`)

Para sustentar a afirmação "EGO economiza avaliações", o pacote `adaptive_3dbp.tuning` compara, com objetivo/instâncias/seeds/orçamento idênticos: Random Search [10] (sanidade), EGO clássico, EGO contextual, **SMAC3** [8] (florestas aleatórias + intensificação; `pip install '.[tuners]'`), **SMAC3 AC** (`smac3_ac`: `AlgorithmConfigurationFacade` com instâncias e features e *racing* entre instâncias) e **irace** [11] (iterated racing em R, via campanha gerada). A campanha também tem dois baselines por instância simples, configuração por classe (`per_class`) e vizinho mais próximo no espaço de features (`nearest_neighbor`), com o orçamento do Random Search. Correções da 0.2.0: o alvo do SMAC3 usa a seed que o SMAC atribui (antes a intensificação repetia runs idênticos); o irace tem o mesmo orçamento dos outros braços (sem o piso de 96 runs) e o incumbente é a elite final do próprio irace (`docs/decisoes_cientificas.md`, seção 4). **Importante (parecer §5.1):** SMAC3, irace, Random Search e EGO clássico resolvem *single best configuration* (uma configuração global); o EGO contextual resolve configuração POR instância em regime zero-shot. São baselines do problema global, não concorrentes diretos — a campanha declara o protocolo de cada braço (`docs/decisoes_cientificas.md`, seção 12).

## 4. Perguntas de Pesquisa

Numeração canônica de `docs/experiment_protocol.md`. As que dizem respeito ao EGO: **RQ4** (o GPR aprende uma relação útil entre features, configuração e desempenho?), **RQ5** (o EGO reduz avaliações de configuração frente ao Random Search?), **RQ6** (a configuração aprendida generaliza fora da distribuição de treino?) e **RQ7** (a política por instância supera a melhor configuração global com o mesmo orçamento?). O custo do configurador em relação ao custo de resolver instâncias é uma métrica de aprendizado do protocolo, não uma RQ separada. A `campaign-632591ae` não mostrou superioridade significativa na RQ7, mas rodou com código anterior à 0.2.0 e precisa ser refeita.

## 5. Frentes Recentes Relacionadas

Além de SMAC3/irace, a área evolui em: Noisy EI por integração Monte Carlo [5] e correções fechadas do EI sob ruído [6]; otimização bayesiana contextual/transfer learning entre instâncias [9]; e, mais recentemente, configuradores baseados em aprendizado profundo e por instância (GNNs e RL para *instance-wise configuration*). O desenho atual (GP exato + EI plug-in) é o ponto de partida defensável e substituível: as interfaces dos configuradores isolam o surrogate (troca por BoTorch [12] sem alterar chamadores).

## Referências

1. Jones, D. R.; Schonlau, M.; Welch, W. J. *Efficient Global Optimization of Expensive Black-Box Functions.* Journal of Global Optimization, 13, 1998.
2. Frazier, P. I. *A Tutorial on Bayesian Optimization.* arXiv:1807.02811, 2018.
3. Picheny, V.; Wagner, T.; Ginsbourger, D. *A benchmark of kriging-based infill criteria for noisy optimization.* Structural and Multidisciplinary Optimization, 48, 2013.
4. Vazquez, E.; et al. *Global optimization based on noisy evaluations: an empirical study of two statistical approaches.* J. Phys.: Conf. Ser., 135, 2008.
5. Letham, B.; Karrer, B.; Ottoni, G.; Bakshy, E. *Constrained Bayesian Optimization with Noisy Experiments.* Bayesian Analysis, 14(2), 2019.
6. Zhou, H.; et al. *A Corrected Expected Improvement Acquisition Function Under Noisy Observations.* PMLR v222 (ACML), 2024.
7. Hutter, F.; Hoos, H. H.; Leyton-Brown, K. *Sequential Model-Based Optimization for General Algorithm Configuration.* LION, 2011.
8. Lindauer, M.; et al. *SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization.* JMLR 23, 2022.
9. Krause, A.; Ong, C. S. *Contextual Gaussian Process Bandit Optimization.* NeurIPS, 2011.
10. Bergstra, J.; Bengio, Y. *Random Search for Hyper-Parameter Optimization.* JMLR 13, 2012.
11. López-Ibáñez, M.; et al. *The irace package: Iterated racing for automatic algorithm configuration.* Operations Research Perspectives, 3, 2016.
12. Balandat, M.; et al. *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.* NeurIPS, 2020.
