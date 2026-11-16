# GPR — Gaussian Process Regression no Projeto

Como o Gaussian Process funciona, como é usado aqui e quais adaptações foram feitas. Implementação: `src/adaptive_3dbp/surrogates/exact_gp.py`.

## 1. O Que É (em uma página)

Um Processo Gaussiano define uma distribuição sobre funções: para qualquer conjunto de entradas, os valores da função têm distribuição normal conjunta, especificada por uma função de média (aqui zero, após normalização) e um **kernel** k(x, x') que codifica similaridade [1]. Condicionando nos dados observados, obtém-se a posterior em forma fechada (eq. 2.22-2.26 de [1]):

```text
μ(x*)  = k*ᵀ (K + Σ_ruído)⁻¹ y          (média preditiva)
σ²(x*) = k(x*,x*) − k*ᵀ (K + Σ_ruído)⁻¹ k*   (variância preditiva)
```

O diferencial sobre regressores pontuais é o segundo termo: o GP devolve **incerteza calibrada**, que é o ingrediente essencial do Expected Improvement no EGO (`docs/ego.md`). A implementação resolve os sistemas por fatoração de Cholesky, como recomendado em [1] (cap. 2, algoritmo 2.1), por estabilidade e custo O(n³) único.

## 2. Para Que o Projeto Usa o GP

O GP é o **surrogate de desempenho do solver** — nunca um gerador de empacotamentos:

```text
x = [features da instância (16)  +  configuração do SA (17)]   ->   y = performance_loss
```

- Features de instância: `n_items`, `volume_ratio`, razões de aspecto, entropia de tamanhos etc. (`features/instance_features.py`).
- Features de configuração: `cooling_rate`, `iterations_per_temperature`, probabilidades dos operadores etc.
- Target: `gap_to_lower_bound + penalidade de falha` (10 se o status não for `feasible`) — a métrica primária é o gap (objetivo do 3D-BPP). Desde a 0.2.0 o tempo de execução não entra no alvo por padrão (só com `runtime_weight > 0` explícito): o antigo desempate `0.01·log1p(runtime)`, medido em parede sob contenção, decidia a maior parte das diferenças entre braços na `campaign-632591ae`. A penalidade por avaliações foi removida para evitar dupla contagem com o orçamento fixo de avaliações (`docs/decisoes_cientificas.md`, seção 11). O gap é medido contra o **lower bound forte** (seção 1 das decisões), o que reduz o ruído estrutural do alvo.

Esse uso — modelar desempenho de algoritmo em função de configuração e instância — é a essência do *model-based algorithm configuration* [4, 5].

## 3. Escolhas de Modelagem e Adaptações

### 3.1 Kernel Matérn 5/2 com ARD

```text
k(x,x') = σ² (1 + √5 r + 5r²/3) exp(−√5 r),   r = ‖(x−x')/ℓ‖
```

com um lengthscale ℓ_d por dimensão (**ARD** — Automatic Relevance Determination). O Matérn 5/2 é o kernel padrão para otimização bayesiana de funções de desempenho: duas vezes diferenciável, sem a suavidade irrealista do RBF [1, 2]; a recomendação vem de [2] e é o default de BoTorch/GPyTorch [7]. ARD permite que o modelo "desligue" dimensões irrelevantes — essencial num espaço de ~33 dimensões em que nem toda feature importa para toda instância.

### 3.2 Hiperparâmetros por máxima verossimilhança marginal (MLE/MAP)

Lengthscales e variância de sinal são otimizados maximizando a log-verossimilhança marginal (eq. 5.8 de [1]) com **gradiente analítico** (eq. 5.9 de [1]; derivada do Matérn 5/2 implementada à mão e validada contra diferenças finitas em teste) e otimizador Adam em log-espaço com multi-start. Um prior log-normal fraco sobre os log-hiperparâmetros transforma MLE em **MAP**, evitando degenerescência com poucos dados — prática padrão [2, 7].

*Por que importa:* lengthscales fixos não fazem ARD de fato; o surrogate fica não-informativo e o EGO degenera em random search caro.

### 3.3 Ruído heterocedástico (e a correção de escala)

O GP aceita uma variância de ruído **por observação** (`fit(x, y, noise)`), somada à diagonal do kernel **apenas no treino** — formulação padrão para observações com ruído não-uniforme [1, 8]. Importante: a variância retornada por `predict` é a **latente** (incerteza sobre f, sem o ruído observacional) — é ela que o EI deve usar; incluir ruído observacional na aquisição faria a busca preferir regiões mais ruidosas (`docs/decisoes_cientificas.md`).

No projeto, o ruído vem de duas fontes: replicações por seed (variância empírica **com shrinkage para a variância global + piso**, pois com 2 seeds a variância amostral tem 1 grau de liberdade — seção 10 das decisões) ou uma estimativa conservadora com piso para execuções únicas (aprendizado online).

**Correção implementada:** os targets são normalizados internamente (z-score), mas o ruído era somado na escala original — ficava ordens de grandeza errado. Agora `noise/σ_y²` (coberto por teste de regressão).

### 3.4 Normalização e estabilidade numérica

Entradas e saídas são normalizadas internamente (média/desvio), com jitter na diagonal — necessário para Cholesky estável e para que um único prior de hiperparâmetros sirva a qualquer escala de dados [1, 7].

### 3.5 Orçamento adaptativo no caminho online

Cada refit custa O(n³) por passo de otimização. No aprendizado online, o orçamento (iterações/restarts) encolhe quando o dataset cresce, mantendo a API responsiva; refits de orçamento cheio pertencem a campanhas offline.

### 3.6 Persistência

`save()/load()` serializam estado completo (Cholesky, α, normalizações, hiperparâmetros) em `.npz` — o modelo treinado é um artefato versionado, registrado em `surrogate_models` (ver `docs/aprendizado_online.md`). O treino online só usa observações da mesma versão `major.minor` do solver (`persistence/sqlite.py`): a 0.2 mudou a energia, a calibração e a perda, e observações da 0.1 não são comparáveis.

## 4. Limites Conhecidos e Evolução

- GP exato escala mal além de poucos milhares de observações; o caminho de evolução é GPyTorch/BoTorch [7] (a interface dos configuradores já foi desenhada para a troca) ou aproximações esparsas/variacionais [9].
- Em dimensão ~33, GP exato é limítrofe; frentes recentes usam kernels aditivos, embeddings e *trust regions* (TuRBO [10]).
- Para ruído heterocedástico aprendido (em vez de fornecido), ver [8].

## Referências

1. Rasmussen, C. E.; Williams, C. K. I. *Gaussian Processes for Machine Learning.* MIT Press, 2006. (Caps. 2, 4 e 5.)
2. Snoek, J.; Larochelle, H.; Adams, R. P. *Practical Bayesian Optimization of Machine Learning Algorithms.* NeurIPS, 2012.
3. Stein, M. L. *Interpolation of Spatial Data: Some Theory for Kriging.* Springer, 1999. (Fundamenta a preferência pelo Matérn.)
4. Hutter, F.; Hoos, H. H.; Leyton-Brown, K. *Sequential Model-Based Optimization for General Algorithm Configuration.* LION, 2011.
5. Lindauer, M.; et al. *SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization.* JMLR, 2022.
6. Frazier, P. I. *A Tutorial on Bayesian Optimization.* arXiv:1807.02811, 2018.
7. Balandat, M.; et al. *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.* NeurIPS, 2020.
8. Griffiths, R.-R.; et al. *Achieving Robustness to Aleatoric Uncertainty with Heteroscedastic Bayesian Optimisation.* Machine Learning: Science and Technology, 2021.
9. Titsias, M. *Variational Learning of Inducing Variables in Sparse Gaussian Processes.* AISTATS, 2009.
10. Eriksson, D.; et al. *Scalable Global Optimization via Local Bayesian Optimization (TuRBO).* NeurIPS, 2019.
