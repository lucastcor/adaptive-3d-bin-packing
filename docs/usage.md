# Guia de uso

Execute os comandos a partir da raiz, com o ambiente virtual ativado e o pacote instalado conforme o README.

## Serviços e core nativo

```bash
./scripts/run_all.sh
./scripts/run_all.sh --no-native
./scripts/run_all.sh --no-frontend
```

Logs e PIDs ficam em `artifacts/logs/` e `artifacts/pids/`. O script permanece em execução; `Ctrl+C` encerra seus processos. O frontend usa `http://127.0.0.1:8000` como API; `VITE_API_BASE` permite alterar esse endereço.

Separadamente:

```bash
./scripts/build_native.sh
python -m uvicorn adaptive_3dbp.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Em outro terminal:

```bash
cd frontend
npm ci
npm run dev
```

O script de build usa C++17, otimização `-O3` e `-ffp-contract=off`. A ponte verifica o hash do fonte, flags e compilador para decidir se recompila. Se o core não estiver disponível, o solver pode fazer fallback para Python; confira `solver` e `diagnostics` no resultado.

## CLI

```bash
adaptive-3dbp --help
adaptive-3dbp validate-instance data/raw/example_20.json
adaptive-3dbp solve --instance data/raw/example_20.json --seed 123
adaptive-3dbp solve --instance data/raw/example_20.json --seed 123 --native
```

Multi-start paralelo com persistência:

```bash
adaptive-3dbp solve --instance data/raw/example_20.json \
  --seed 123 --parallel-mode process_pool --parallel-workers 4 \
  --multi-start-count 8 --convergence-sample-rate 5 --persist
```

O orçamento de avaliações é compartilhado pelos starts; número de workers não deve ser interpretado como orçamento extra. O modo cooperativo está disponível como `cooperative_process_pool` e não é determinístico.

## Planilhas

```bash
adaptive-3dbp spreadsheet-template --out data/raw/instance_template.xlsx
adaptive-3dbp import-spreadsheet minha_instancia.xlsx --out instancia.json
adaptive-3dbp solve --instance minha_instancia.xlsx --seed 123
```

O formato contém abas `meta`, `bin`, `items` e `min_distance` opcional. Na aba `items`, `allowed_rotations` aceita vazio (todas as rotações ortogonais), `fixed` ou uma lista de IDs. A interface permite baixar o modelo, importar a planilha e exportar a solução com caixa, coordenadas e dimensões posicionadas.

## API

Confira os contratos completos no [Swagger local](http://127.0.0.1:8000/docs).

Exemplo com a instância versionada, sem persistir:

```python
import json
from pathlib import Path
from urllib.request import Request, urlopen

instance = json.loads(Path("data/raw/example_20.json").read_text())
payload = {"instance": instance, "config": {"max_evaluations": 1000},
           "seed": 123, "persist": False}
request = Request("http://127.0.0.1:8000/api/solve",
                  data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request) as response:
    result = json.load(response)["result"]
print(result["status"], result["best_found_bins"], result["lower_bound"])
```

| Rotas | Finalidade |
|---|---|
| `GET /health` | Saúde da API e core nativo |
| `GET /api/config/default` | Configuração padrão |
| `POST /api/validate` | Validar instância |
| `POST /api/solve` | Execução síncrona |
| `POST /api/solve-jobs` | Iniciar execução assíncrona |
| `GET /api/solve-jobs/{id}` | Progresso e resultado |
| `POST /api/solve-jobs/{id}/pause`, `/resume`, `/cancel` | Controle, conforme suporte do modo |
| `GET /api/runs`, `/api/runs/{id}/bundle` | Histórico e reabertura |
| `GET /api/learning/status` | Estado do aprendizado |
| `POST /api/config/suggest` | Sugestão via GPR/EI ou aquecimento aleatório |
| `POST /api/campaigns` | Campanha experimental |

`persist=true` é o padrão na API/interface. Os dados ficam em `artifacts/experiments.db`; modelos são gerados em `artifacts/models/`. A distribuição pública não inclui bancos nem modelos pessoais. O GPR é reajustado após execuções persistidas quando há observações suficientes; observações de outra versão `major.minor` não são misturadas. Veja [aprendizado online](aprendizado_online.md).

## Parallel tempering

Selecione na configuração enviada à API:

```json
{
  "algorithm": "parallel_tempering",
  "max_evaluations": 3000,
  "pt_replicas": 8,
  "pt_swap_interval": 10,
  "pt_min_temperature_ratio": 0.01,
  "pt_max_temperature": null
}
```

Esse algoritmo roda em Python; não há seletor na CLI `solve` nem na interface. Uma solicitação de core nativo faz fallback com diagnóstico. Detalhes em [Simulated Annealing](simulated_annealing.md).

## Campanhas e tuners opcionais

```bash
python -m pip install -e ".[dev,tuners,plots]"
adaptive-3dbp run-campaign --spec configs/campaigns/pilot.json
adaptive-3dbp run-campaign --spec configs/campaigns/mpv_pilot.json
```

SMAC3 é opcional. irace requer R e o pacote R `irace`; instale-os separadamente. Braços indisponíveis podem aparecer como `skipped`; não os conte como comparações realizadas.

As specs controlam dataset, splits, tuners, seeds, trials, avaliações e core. Para MPV use `dataset.benchmark`; para core nativo, `budget.native_core: true`. A interface não expõe todas as opções das specs; use CLI/API para MPV e core da campanha. Saídas são geradas em `artifacts/campaigns/<id>/` e ficam fora do Git.

Antes de uma campanha confirmatória: finalize e versione as alterações, fixe ambiente/core, pré-especifique comparações e separe treino/validação/teste. A preparação do README e a demonstração não substituem essa campanha. Leia [o protocolo](experiment_protocol.md).
