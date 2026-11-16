# Arquitetura Implementada

A aplicação está organizada em camadas para separar domínio, busca, execução, persistência e interface.

## Camadas

1. **Domain:** contratos Pydantic, rotações, bounds, scoring (energia `E = b + r`) e validação geométrica (colisão, rotações permitidas, `min_distance`, peso e apoio).
2. **Features:** extração de descritores numéricos da instância para análise e GPR.
3. **Constructors:** soluções iniciais por volume, maior dimensão, maior face e aleatoriedade com seed.
4. **Decoders:** Extreme Points determinístico para converter ordem/rotação em placements 3D, respeitando peso e apoio.
5. **Metaheuristics:** Simulated Annealing modular com operadores de vizinhança e parallel tempering (`algorithm="parallel_tempering"`, só Python) sobre a mesma vizinhança, decoder e energia.
6. **Native:** ponte Python para o binário C++ standalone.
7. **Parallelism:** multi-start independente por processos Python ou subprocessos C++.
8. **Persistence:** SQLite para execuções, placements, observações de aprendizado e modelos surrogate.
9. **Learning:** criação de observações e treino online do GPR.
10. **Configurators:** Random Search, EGO clássico e EGO contextual com ruído.
11. **Tuning e Experiments:** objetivo compartilhado, SMAC3 (`smac3`, `smac3_ac`), irace, baselines por instância, runner de campanha, geração de dataset e estatística.
12. **Benchmarks:** gerador bit-exato das instâncias de Martello, Pisinger e Vigo (`benchmarks/mpv.py`).
13. **API/Frontend:** FastAPI e React/Three.js.

## Fluxo De Execução

```text
HTTP/UI/CLI
  -> valida ProblemInstance
  -> resolve via Python SA ou C++ SA
  -> valida placements finais
  -> calcula métricas
  -> salva run quando persist=true
  -> cria learning_observation
  -> treina GPR online quando há dados suficientes
  -> retorna solução, diagnósticos e convergência
```

## Decisões Principais

- O SA opera sobre representação indireta: permutação de itens e vetor de rotações.
- O decoder é responsável por produzir soluções geometricamente válidas.
- O validador final roda no backend mesmo quando a solução veio do C++.
- O decoder do core C++ é bit a bit idêntico ao Python (rotações permitidas, `min_distance`, peso, apoio e política de posicionamento), verificado por `tests/unit/test_native_parity.py` (a paridade é com o Python 3.12+; no 3.11 há diferenças raras em empates de ponto flutuante); novas restrições precisam ser portadas e cobertas por esse teste.
- O protocolo Python/C++ é textual (versão `v2`, campos nomeados) e pequeno para evitar acoplamento com CMake, pybind11 ou ABI específica. O Python manda o próprio pool inicial.
- O binário C++ é recompilado quando o SHA-256 de `cpp/sa_core.cpp` não bate com o registrado em `artifacts/bin/sa_core.sha256` (não por data de modificação).
- O paralelismo usa granularidade grossa por multi-start para reduzir comunicação e locks.
- O GPR modela desempenho de configuração, não coordenadas de empacotamento.
- O aprendizado online é rastreável porque cada observação referencia uma execução salva.

## Contrato De Convergência

Cada solver retorna `convergence` com pontos amostrados. O campo suporta análise anytime:

- melhor objetivo por avaliação;
- objetivo corrente;
- bins usados;
- taxa de aceite;
- eventos de melhoria;
- efeito de temperatura e operadores.

`convergence_sample_rate` controla a quantidade de pontos salvos.

## Persistência

Tabelas principais:

- `experiment_runs`: metadados e JSON completo de instância, config e resultado.
- `placements`: placements normalizados para consulta.
- `learning_observations`: vetor de entrada do GPR, target e variância de ruído.
- `surrogate_models`: artefatos e métricas do GPR treinado.

Essa estrutura permite reabrir uma execução por `/api/runs/{run_id}/bundle` e reproduzir visualização, curva e configuração.
