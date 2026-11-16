# Estado da distribuição

Preparação em 16/11/2026: repositório novo, com histórico independente, inicialmente privado para revisão. A abertura pública e a escolha da licença permanecem pendentes de decisão do autor.

## Conteúdo

Esta árvore contém fontes, testes e fixtures do projeto, exemplos pequenos, documentação técnica, mídias selecionadas e CI. Não contém artigos em PDF, documentos Word, rascunhos de manuscritos, vault pessoal, bancos, modelos treinados, campanhas históricas ou o histórico Git do projeto anterior. A legenda do LinkedIn foi mantida fora da árvore versionada.

O código tem como base a versão 0.3.0 do Adaptive 3D Bin Packing. A publicação começa com um commit raiz próprio; commits de campanhas anteriores não são importados. Metadados científicos antigos mencionados na documentação são registros de procedência, não resultados de uma campanha executada neste repositório.

## Licença e procedência

Nenhuma licença de redistribuição foi escolhida. O repositório privado foi preparado para revisão e não é anunciado como software open source já licenciado.

O módulo `src/adaptive_3dbp/benchmarks/mpv.py` declara implementar independentemente o procedimento de geração MPV. O [código C original](https://hjemmesider.diku.dk/~pisinger/new3dbpp/readme.3dbpp) restringe seu uso a pesquisa e fins acadêmicos. Esse C original não é distribuído aqui. Antes de aplicar uma licença ao conjunto, é necessário confirmar a procedência da reimplementação e preservar os avisos aplicáveis de componentes de terceiros. Uma declaração no fonte não substitui essa conferência.

Referências científicas são mantidas como citações e links. Os agradecimentos à contribuição de Alysson estão em `ACKNOWLEDGMENTS.md`.

## Verificação

Validação local em 16/11/2026: 224 testes passaram, sem skips, com core C++ obrigatório e Python 3.14.3 (1 aviso de depreciação de dependência), em aproximadamente 72 s. A instalação limpa e o build do frontend passaram; npm reportou zero vulnerabilidades conhecidas. Os links locais e a seleção de arquivos foram conferidos. As campanhas confirmatórias da versão 0.3.0 continuam pendentes; a captura em vídeo não substitui os experimentos.

A CI verifica Python 3.12/3.13 e Node 22 após o push. A varredura automatizada de formatos comuns de credenciais é limitada; não é uma certificação jurídica ou de segurança.
