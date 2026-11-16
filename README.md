# Adaptive 3D Bin Packing

**Do conjunto de itens à solução em 3D: heurísticas, busca estocástica e experimentos reproduzíveis.**

Plataforma experimental para o **3D Bin Packing offline**: posicionar itens retangulares em caixas idênticas, sem sobreposição, buscando reduzir o número de caixas. Implementação em Python com core C++ opcional, API FastAPI e interface React/Three.js.

[Demonstração](#demonstração) · [Executar](#executar-localmente) · [Como funciona](#como-funciona) · [Estado científico](docs/status.md) · [Guia de uso](docs/usage.md)

## Demonstração

![Demonstração da interface e da solução 3D](docs/assets/demo/packing-demo.gif)

[Assistir à demonstração em MP4](docs/assets/demo/packing-demo.mp4) · [Mídias e reprodução da captura](docs/media.md)

A captura mostra uma execução real da versão 0.3.0, com 8 buscas em 4 workers C++, e a inspeção da solução. É uma demonstração de funcionamento; uma instância isolada não mede desempenho em benchmarks.

## O que você pode explorar

- **Empacotamento 3D:** navegar pelas caixas, inspecionar os itens e acompanhar a curva de busca.
- **Busca:** Simulated Annealing sobre ordem e rotações, decodificado por Extreme Points com first fit e projeção dos pontos.
- **Restrições:** rotações ortogonais permitidas, distância mínima entre pares, peso total por caixa e proporção mínima de apoio da base.
- **Execução:** core Python ou C++, multi-start e processos paralelos. Parallel tempering e cooperação entre workers estão disponíveis no core Python.
- **Dados:** importar instâncias e exportar soluções em planilhas, salvar execuções em SQLite e reabrir resultados.
- **Configuração de algoritmos:** GPR, EGO clássico/contextual, Random Search e integração opcional com SMAC3 e irace.

## Executar localmente

Requisitos: **Python 3.11+**, **Node.js 20.19+ ou 22.12+** e npm. Para comparar o decoder Python/C++ bit a bit, use Python **3.12+**. O core nativo requer um compilador com C++17 (`g++`).

```bash
git clone https://github.com/lucastcor/adaptive-3d-bin-packing.git
cd adaptive-3d-bin-packing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cd frontend
npm ci
cd ..
./scripts/run_all.sh
```

Abra [a interface](http://127.0.0.1:5173) e clique em **Executar**. A [documentação interativa da API](http://127.0.0.1:8000/docs) fica na porta 8000. `Ctrl+C` encerra os serviços. Para executar somente com Python: `./scripts/run_all.sh --no-native`.

Pela CLI, sem iniciar a interface:

```bash
adaptive-3dbp validate-instance data/raw/example_20.json
adaptive-3dbp solve --instance data/raw/example_20.json --seed 123
# Acrescente --native para selecionar o core C++.
```

O [guia de uso](docs/usage.md) cobre API, planilhas, paralelismo, campanhas e dependências opcionais.

## Como funciona

```mermaid
flowchart LR
    A[Instância e restrições] --> B[Ordem dos itens e rotações]
    B --> C[Extreme Points]
    C --> D[Validação geométrica e score]
    D --> E[Simulated Annealing]
    E --> B
    D --> F[Solução, curva e visualização 3D]
    F --> G[SQLite e observações]
    G --> H[GPR e configuração via EGO]
    H -. próximas execuções .-> E
```

O SA modifica a representação combinatória; o decoder escolhe as posições. O GPR aprende a relação entre instância, configuração e perda observada. Ele **não prevê diretamente coordenadas** nem garante melhoria em cada execução.

## Estado do projeto — 0.3.0

O 3D-BPP é um problema de otimização **NP-difícil**. Este projeto usa heurísticas e metaheurísticas; não oferece garantia geral de ótimo. Quando uma solução válida atinge um limite inferior válido, a contagem de caixas está certificada para aquela instância.

Na 0.3.0, o decoder passou a usar first fit entre caixas e projeção dos pontos extremos. Há testes de regressão contra três ordenações EP-FFD do benchmark MPV, com tolerância por classe. Isso valida parte do comportamento do decoder; **não demonstra superioridade sobre a literatura**.

**As campanhas históricas foram executadas com versões anteriores. Ainda falta uma campanha confirmatória com a 0.3.0.** Não há evidência atual para anunciar ganho do EGO contextual, speedup quantificado ou estado da arte. Veja a [análise técnica e científica](docs/status.md) e o [protocolo experimental](docs/experiment_protocol.md).

Limitações relevantes:

- Apoio de base e peso total não substituem simulação física: faltam centro de massa e carga suportada por item.
- O lower bound pode ser frouxo; gap até esse limite não é distância conhecida ao ótimo.
- O decoder Python/C++ tem testes de paridade; as trajetórias do SA usam geradores aleatórios distintos.
- O modo cooperativo é não determinístico. Parallel tempering e cooperação fazem fallback para Python.
- A API foi feita para execução local e não possui autenticação ou isolamento entre usuários.

## Verificação

```bash
source .venv/bin/activate
./scripts/build_native.sh
ADAPTIVE_3DBP_REQUIRE_NATIVE=1 python -m pytest
cd frontend
npm run build
```

Os testes de paridade usam fixtures versionadas em `tests/fixtures/`; não dependem de campanhas ou bancos pessoais. O [workflow de CI](.github/workflows/ci.yml) verifica backend, core nativo e build do frontend.

## Estrutura

```text
src/adaptive_3dbp/   domínio, decoder, busca, GPR, campanhas, API e CLI
cpp/                core C++17
frontend/           React, TypeScript e Three.js
configs/            espaços de busca e especificações de campanha
data/raw/           instâncias pequenas para começar
tests/              testes unitários, integração e fixtures
scripts/            execução, build, diagnósticos e preparação pública
docs/               documentação técnica e mídias selecionadas
```

Artigos baixados, rascunhos, resumos pessoais, bancos e saídas de campanhas não fazem parte da distribuição pública. As referências são mantidas como citações e links.

## Documentação e contribuições

| Documento | Conteúdo |
|---|---|
| [Uso](docs/usage.md) | CLI, API, planilhas e campanhas |
| [Estado científico](docs/status.md) | Evidências, limites e próximos experimentos |
| [Arquitetura](docs/architecture.md) | Camadas e fluxo de execução |
| [Simulated Annealing](docs/simulated_annealing.md) | Energia, temperatura e operadores |
| [GPR](docs/gpr.md) / [EGO](docs/ego.md) | Surrogate e configuração de algoritmos |
| [Aprendizado por execução](docs/aprendizado_online.md) | Persistência e atualização do GPR |
| [Paralelismo cooperativo](docs/paralelismo_cooperativo.md) | Coordenação e limites de determinismo |
| [Benchmark MPV](docs/benchmark_mpv.md) | Geração, orientação e separação dos dados |
| [Decisões científicas](docs/decisoes_cientificas.md) | Justificativas e correções |
| [Referências](docs/references.md) | Fontes primárias |

Para contribuir, leia [CONTRIBUTING.md](CONTRIBUTING.md). A definição da licença de distribuição está pendente; consulte [o estado da preparação pública](docs/public-release.md).

## Origem e agradecimentos

O projeto nasceu do interesse por otimização combinatória desenvolvido em um desafio de Bin Packing 1D, realizado por Lucas e Alysson durante a graduação. A exploração de SA e GRASP naquele desafio motivou estudos posteriores e esta implementação em 3D.

Alysson também contribuiu para a paralelização do algoritmo e do pipeline. Veja [os agradecimentos](ACKNOWLEDGMENTS.md).
