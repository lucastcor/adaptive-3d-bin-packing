# Aprendizado Online: Fluxo, Banco de Dados e Operação

Este documento explica como o aprendizado online funciona na prática: onde os dados ficam, o que acontece a cada execução, o que acontece ao reiniciar, e como consultar tudo — tanto para operar a aplicação quanto para quem for auditar a pesquisa.

**Terminologia (precisão científica).** "Online" aqui descreve o *fluxo operacional* — acumulação sequencial de dados e reajuste periódico em lote a partir do histórico persistente. O modelo é um **GP reajustado sequencialmente** (sequentially refit GP): a cada retreino, o ajuste é refeito sobre toda a tabela. NÃO se trata de atualização incremental online da fatoração do GP (rank-1 updates da Cholesky) — essa distinção é declarada também no artigo.

## 1. Visão Geral

A cada execução persistida do solver, o sistema grava a evidência experimental (instância, configuração, resultado) e retreina um surrogate GPR que aprende a relação:

```text
features da instância + configuração do solver  ->  perda esperada (gap + penalidade de falha)
```

O GPR **não monta caixas**: ele aprende o *desempenho* do solver, acumulando conhecimento experimental execução a execução. Detalhes do modelo em `docs/gpr.md`; o uso desse conhecimento para sugerir configurações em `docs/ego.md`.

## 2. O Banco: Onde Fica, Precisa "Subir"?

**Não há servidor de banco para subir.** O banco é **SQLite embutido** — um único arquivo:

```text
artifacts/experiments.db
```

- O arquivo é criado automaticamente na primeira execução (com `CREATE TABLE IF NOT EXISTS`); não existe passo de migração manual.
- O banco vive **dentro do processo do backend** (a biblioteca SQLite abre o arquivo direto); não há porta, usuário ou senha.
- **No Docker:** o `docker-compose.yml` monta `../artifacts` como volume em `/app/artifacts`. Ou seja, o `.db` e os modelos treinados ficam no **host**, fora do container — derrubar/recriar o container não perde nada.
- Modelos treinados (artefatos do GPR) ficam ao lado: `artifacts/models/online_gpr_run_{run_id}.npz/.json` e `online_gpr_latest.npz/.json`.

A escolha por SQLite é deliberada para a fase de pesquisa: zero operação, arquivo único versionável/copiável, transações ACID locais e leitura direta por qualquer ferramenta (pandas, DBeaver, `sqlite3` CLI). Migrar para Postgres só se justificaria com escrita concorrente multi-usuário.

## 3. Esquema

| Tabela | Conteúdo | Papel |
|---|---|---|
| `experiment_runs` | instância, config, seed, solver, métricas e resultado completo (JSONs íntegros) | evidência primária, imutável |
| `placements` | posição de cada item por execução | reconstrução/auditoria geométrica |
| `learning_observations` | vetor `x` (features+config), `target`, `noise_variance` | dataset do GPR (1 por run, `UNIQUE(run_id)`) |
| `surrogate_models` | tipo, caminho do artefato, nº de observações, métricas | rastreabilidade de cada modelo treinado |

Imutabilidade: execuções concluídas **nunca são sobrescritas** — novas tentativas geram novos `run_id`. Isso preserva a trilha de evidência exigida por boas práticas de benchmarking experimental [5, 6].

## 4. O Que Acontece A Cada Execução

Com `persist=true` (padrão da API e do frontend), o fluxo do `POST /api/solve` é:

```text
1. valida instância
2. resolve (SA + Extreme Points, core Python ou C++)
3. save_run:
   INSERT em experiment_runs (instância/config/resultado completos)
   INSERT em placements (cada item posicionado)
4. observe_and_train:
   a. build_learning_observation:
      x      = 16 features da instância + 17 features da configuração
      target = gap_to_lower_bound
               + penalidade de falha (10.0 se não-feasible)
               (desde a 0.2.0 o tempo de execução não entra no target, salvo
                runtime_weight > 0 explícito; a penalidade por avaliações foi
                removida: com orçamento fixo de avaliações, penalizá-las
                contaria o recurso duas vezes — docs/decisoes_cientificas.md,
                seção 11)
      noise  = variância estimada da observação (execução única é ruidosa);
               no treino aplica-se piso proporcional à variância média
   b. INSERT em learning_observations
   c. train_online_gpr:
      SELECT as observações da mesma versão major.minor do solver
             (até 5000, ordem cronológica; runs da 0.1 ficam de fora)
      se < 3 observações: retorna "aquecendo", não treina
      senão: ajusta GPR (hiperparâmetros por MLE/MAP, orçamento adaptativo
             ao tamanho do dataset — ver docs/gpr.md)
      salva online_gpr_run_{id}.npz/.json e atualiza online_gpr_latest.*
      INSERT em surrogate_models
5. resposta inclui o status do aprendizado nos diagnostics
```

A CLI com `--persist` executa exatamente o mesmo caminho.

**Custo:** o retreino é O(n³) no número de observações; o orçamento de otimização de hiperparâmetros encolhe automaticamente quando o dataset cresce, mantendo a API responsiva. Refits com orçamento cheio pertencem a campanhas offline.

## 5. E Se Eu Fechar e Subir De Novo?

Nada se perde e nada precisa ser refeito manualmente:

- **Todo o estado está em disco** (`experiments.db` + `artifacts/models/`). O processo do backend não guarda estado de aprendizado em memória entre requisições.
- Ao reiniciar, o repositório reconecta ao arquivo e os `CREATE TABLE IF NOT EXISTS` são no-ops.
- O treino é **reconstruído do banco a cada execução**: o modelo é uma função pura da tabela `learning_observations` (full refit, não incremental). Consequência importante para reprodutibilidade: dado o mesmo banco, o mesmo modelo é reproduzível; não há estado oculto acumulado no processo.
- Se o arquivo `.db` for apagado, o sistema recomeça do zero ("aquecendo" até 3 observações) — por isso o banco deve ser tratado como artefato de pesquisa: copie/versione `artifacts/experiments.db` junto com os resultados.

## 6. Como Consultar

Pela API:

```text
GET /api/learning/status      -> nº de observações + último modelo treinado
GET /api/runs                 -> execuções recentes
GET /api/runs/{id}            -> resultado completo de uma execução
GET /api/runs/{id}/bundle     -> instância + config + resultado (reanálise)
```

Direto no arquivo (qualquer ferramenta SQLite):

```sql
-- evolução do target de aprendizado no tempo
SELECT run_id, target, noise_variance, created_at
FROM learning_observations ORDER BY observation_id;

-- histórico de modelos treinados
SELECT model_id, observation_count, created_at FROM surrogate_models;
```

Para carregar o modelo treinado em Python:

```python
from adaptive_3dbp.surrogates.exact_gp import ExactGaussianProcess
gp = ExactGaussianProcess.load(Path("artifacts/models/online_gpr_latest.npz"))
```

## 7. Posição Metodológica

Aprender online a cada chamada tem um risco conhecido: dados atípicos ou seeds sortudas contaminam o modelo e dificultam reprodutibilidade. O projeto mitiga isso com três propriedades: (1) a evidência bruta é imutável e completa — o modelo pode sempre ser reconstruído ou re-treinado offline com curadoria; (2) cada observação aponta para um `run_id` concreto com seed, config e resultado; (3) a separação train/validation/test para *avaliação científica* do surrogate segue o protocolo de `docs/experiment_protocol.md`, com o conjunto de teste usado uma única vez. O modelo online deve ser visto como o "modelo operacional"; conclusões de pesquisa exigem campanhas controladas [5, 6].

Esse desenho segue a prática de *algorithm configuration* com dados persistentes [4, 5] e a literatura de otimização bayesiana com avaliações ruidosas [2, 3].

## Referências

1. Rasmussen, C. E.; Williams, C. K. I. *Gaussian Processes for Machine Learning.* MIT Press, 2006.
2. Snoek, J.; Larochelle, H.; Adams, R. P. *Practical Bayesian Optimization of Machine Learning Algorithms.* NeurIPS, 2012.
3. Letham, B.; Karrer, B.; Ottoni, G.; Bakshy, E. *Constrained Bayesian Optimization with Noisy Experiments.* Bayesian Analysis, 14(2), 2019.
4. Hutter, F.; Hoos, H. H.; Leyton-Brown, K. *Sequential Model-Based Optimization for General Algorithm Configuration.* LION, 2011.
5. López-Ibáñez, M.; et al. *The irace package: Iterated racing for automatic algorithm configuration.* Operations Research Perspectives, 2016.
6. Eggensperger, K.; Lindauer, M.; Hutter, F. *Pitfalls and Best Practices in Algorithm Configuration.* JAIR, 64, 2019.
