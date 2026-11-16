# Mídias da demonstração

Captura feita em **14/11/2026**, com a aplicação e o solver reais. Não havia GIFs nem gravações no diretório do projeto ou nos caminhos de mídia encontrados no histórico local; havia apenas screenshots antigos. Os arquivos abaixo foram produzidos nesta preparação.

| Arquivo em `docs/assets/demo/` | Uso | Formato |
|---|---|---|
| `packing-demo.gif` | Prévia no README | 9 s, 880 × 496, 10 fps, cerca de 0,98 MiB |
| `packing-demo.mp4` | Gravação da interface | 31,64 s, 1920 × 1080, H.264, cerca de 3,24 MiB |
| `linkedin-demo.mp4` | Vídeo para anexar ao post | Mesmo corte limpo, sem textos adicionados |
| `app-desktop.png` | Screenshot/capa | Captura da página após a execução |
| `demo-run.json` | Proveniência | Configuração, resultado e resumo da observação de processos |

Os MP4 usam `yuv420p` e metadados `faststart`. A versão final mostra somente a interface: sem cartelas, títulos, legendas sobrepostas, música ou narração. O carregamento inicial foi cortado; a execução e as interações mantêm o tempo real. O GIF é um trecho da inspeção 3D em loop.

## Execução filmada

- Instância: `data/raw/demo_60.json`, 60 itens sintéticos; geração com `random.Random(42)`, três dimensões inteiras em `[16, 46]` por item, na ordem comprimento/largura/altura; caixa de lado 100, rotações ortogonais livres.
- Solver 0.3.0, core C++, seed 123, **Pool com 4 workers e 8 buscas**, teto total de 96.000 avaliações (12.000 por busca) e limite de tempo configurado de 30 s; amostragem de convergência 1.
- Resultado: solução válida em **3 caixas**, lower bound **2**, `optimal_proven=false`; **21.310 avaliações** somadas entre as buscas, aproximadamente **5,65 s** de parede. Cada busca pode encerrar antes do teto por seus critérios de parada.
- O resultado identifica `simulated_annealing_cpp_core_parallel` e registra `workers=4, starts=8, evaluations_per_start=12000`, sem fallback para o core Python.
- A observação dos processos do sistema operacional, a cada 40 ms, identificou **8 subprocessos nativos distintos**, até **4 simultâneos**, com crescimento de CPU em quatro processos no mesmo intervalo de amostragem. Foram observados cerca de 17,78 s de CPU somados. Isso confirma a atividade dos workers além da configuração exibida na tela; não é uma medição de speedup.
- `persist=false`, com SQLite isolado em `.private/media-tools/demo.db`. Nenhum endpoint de solução foi simulado; a resposta HTTP final foi registrada e não houve erro de JavaScript.
- A curva retornada pelo pool C++ é a da busca vencedora, enquanto o total de avaliações soma as oito buscas.

O desempenho de uma instância sintética não é um benchmark. O contexto científico fica na legenda do post e na documentação, conforme preferência do autor. A captura final usa Vite 7.3.8. A gravação sequencial anterior foi preservada localmente em `.private/media-tools/previous-sequential/`.

## Reproduzir a captura

Além das dependências normais, instale FFmpeg no sistema e Playwright em uma área local ignorada:

```bash
npm install --prefix .private/media-tools --no-save playwright@1.62.1
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.private/pw-browsers"
node .private/media-tools/node_modules/playwright/cli.js install chromium ffmpeg
```

Em um terminal, a partir da raiz, inicie uma API com banco isolado:

```bash
PYTHONPATH=src python -c 'import uvicorn; import adaptive_3dbp.api.main as api; from adaptive_3dbp.persistence.sqlite import SQLiteExperimentRepository; api.repository = SQLiteExperimentRepository(".private/media-tools/demo.db"); uvicorn.run(api.app, host="127.0.0.1", port=8000)'
```

Em outro terminal, inicie o frontend (`cd frontend; npm ci; npm run dev`). Com ambos disponíveis e o core nativo compilado:

```bash
export PLAYWRIGHT_MODULE="$PWD/.private/media-tools/node_modules/playwright"
node scripts/record_demo.cjs
python scripts/render_demo.py
```

`CHROME_PATH` permite escolher um executável do Chrome já instalado; `DEMO_URL` permite alterar a URL da interface. O tempo da execução depende do hardware.

Os scripts substituem as mídias públicas da demonstração. A captura bruta WebM, a observação dos processos e a contact sheet ficam em `.private/media-tools/`. Não inclua essa pasta no repositório público. A captura usa a [gravação de vídeo do Playwright](https://playwright.dev/docs/videos), encerrando o contexto antes de usar o arquivo final.
