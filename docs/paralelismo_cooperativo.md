# Paralelismo Cooperativo com Memória Central

Este documento explica **por que** o projeto adota paralelismo cooperativo, **como** ele foi implementado e quais decisões de sistemas distribuídos foram tomadas (e quais foram deliberadamente adiadas). Propostas de pesquisa e rascunhos ficam no arquivo privado; este documento descreve a implementação pública.

## 1. Contexto e Motivação

O solver executa Simulated Annealing multi-start: cada worker recebe uma solução inicial distinta e percorre uma cadeia de busca independente; um agregador escolhe a melhor solução ao final. Esse desenho (modo `process_pool`) maximiza isolamento e reprodutibilidade, mas desperdiça informação: um worker que encontra uma solução com uma caixa a menos não comunica a descoberta aos demais, que seguem explorando regiões dominadas do espaço de busca.

O modo cooperativo (`cooperative_process_pool`) ataca isso com o padrão **central memory / solution warehouse** da literatura de metaheurísticas paralelas [1, 2]: os workers trocam apenas a melhor solução global através de uma memória compartilhada, sem nenhum outro estado compartilhado fino. A taxonomia de Crainic classifica isso como busca cooperativa multi-thread com comunicação indireta por memória — historicamente a família com melhores resultados em problemas combinatórios difíceis [1].

Não confundir com o **parallel tempering** (`algorithm="parallel_tempering"`, `docs/simulated_annealing.md`, seção 3.8): lá as réplicas rodam intercaladas num único processo, em temperaturas fixas, e trocam configurações entre temperaturas vizinhas por um critério de Metropolis; é determinístico dado a seed e divide o mesmo orçamento de avaliações. O modo cooperativo é paralelismo de processos com troca da melhor solução global.

## 2. Como Funciona

```text
worker 0 ──┐                    ┌── worker 1
           │   memória central  │
  publica ─┼──> best_score      │
           │    best_order      <── adota se melhor
           │    best_rotations  │
           │    version (lógico)│
  heartbeat┼──> heartbeat_0..N <┼── heartbeat
           │                    │
           └── coordenador (processo pai) monitora e agrega
```

A cada `cooperative_sync_interval` avaliações (padrão 25), cada worker:

1. **emite heartbeat** — grava `time.time()` em sua chave na memória central;
2. **publica** — sob lock, compara seu melhor local com o melhor global (ordem lexicográfica: caixas, fragmentação, −compactação; o volume vazio total saiu da chave por ser determinado pelo número de caixas) e o substitui se for estritamente melhor, incrementando o contador `version`;
3. **adota** — lê a versão global; se for nova e o candidato global for estritamente melhor que o seu melhor local, re-decodifica o candidato (permutação + rotações) com o decoder Extreme Points e continua a busca a partir dele. A re-decodificação é contabilizada como uma avaliação e a solução adotada passa pelo mesmo validador geométrico de qualquer outra — a cooperação nunca contorna a validação.

Configuração:

```json
{
  "parallel_mode": "cooperative_process_pool",
  "parallel_workers": 4,
  "multi_start_count": 8,
  "cooperative_sync_interval": 25,
  "heartbeat_timeout_seconds": 15.0
}
```

Arquivos: `src/adaptive_3dbp/metaheuristics/simulated_annealing/cooperative.py` (canal) e `solver.py` (método `_solve_cooperative` e pontos de sincronização dentro de `_run_single_start`).

## 3. Decisões de Sistemas Distribuídos

### 3.1 Sincronismo de relógio: desnecessário (sequence number monotônico)

A única pergunta que um worker precisa responder é *"essa informação é nova para mim e é melhor que a minha?"* — uma questão de **ordenação causal**, não de tempo absoluto. A implementação usa um **sequence number monotônico** (contador `version` incrementado sob lock). A analogia com relógios lógicos de Lamport [3] é apenas conceitual: com memória central e lock, um contador único é estritamente mais simples que o mecanismo de Lamport (que ordena eventos *sem* coordenação central) e não deve ser chamado de equivalente. Workers comparam versões, nunca timestamps. Por isso, sincronização de relógio físico (NTP, algoritmo de Berkeley) não é necessária: em host único todos os processos já compartilham a mesma fonte de relógio e, mesmo em cenário distribuído, a ordenação lógica seria suficiente para este padrão de troca. Timestamps físicos aparecem apenas nos heartbeats, onde tolerância de segundos torna deriva de relógio irrelevante.

### 3.2 Falhas na versão atual: heartbeat e degradação graciosa

O coordenador é o **processo pai, fixo por construção**: cria os workers, hospeda a memória central (`multiprocessing.Manager`) e agrega resultados. Se ele morre, a computação morre junto — não existe cenário em host único no qual os workers sobrevivam ao coordenador e precisem substituí-lo. Por isso, nesta versão, o tratamento de falhas se resume a falha parcial de worker, em dois níveis: worker que lança exceção vira diagnóstico no resultado (contribuição descartada); worker sem heartbeat há mais de `heartbeat_timeout_seconds` é reportado como travado e ignorado. Em ambos os casos a busca continua com os workers restantes, com um deadline global no coordenador como salvaguarda final.

### 3.3 Cenário distribuído futuro: onde eleição de coordenador entra em cena

O contexto muda quando esta frente evoluir para um **sistema distribuído de verdade: o SA cooperativo rodando em vários computadores (nós)**, cada nó com um ou mais workers, e a memória central virando um serviço de rede hospedado em um nó coordenador. Aí surgem falhas que não existem em host único: o nó coordenador pode cair ou ficar inacessível **enquanto os demais nós continuam vivos com trabalho útil em andamento**. Sem substituição automática, a queda do coordenador paralisa toda a busca — um ponto único de falha que um sistema distribuído precisa eliminar. É esse o problema que algoritmos de eleição de coordenador resolvem.

O clássico é o **algoritmo do valentão** (bully, Garcia-Molina [4]): ao detectar coordenador sem resposta, o nó vivo de maior identificador assume. É simples, mas limitado para o nosso caso: assume detecção de falha confiável (timeout não distingue nó morto de rede lenta), não trata **partições de rede** — duas metades podem eleger dois coordenadores (*split-brain*), corrompendo a memória central com duas "melhores soluções globais" divergentes — e não diz como o estado compartilhado sobrevive à troca.

A solução verificada e adotada pela prática é usar **consenso com eleição de líder integrada**: o **Raft** [5] elege por termos com maioria (quóruns impedem split-brain por construção), replica um log que garante que o novo líder tem todo o estado confirmado (a melhor solução global sobrevive à troca) e é amplamente verificado e usado em produção (etcd/Kubernetes, Consul, TiKV). Paxos [8] dá as mesmas garantias, mas é notoriamente mais difícil de implementar — a motivação explícita do Raft foi ser compreensível.

Tradução para o projeto: na versão multi-nó, a memória central (melhor solução + versão lógica) seria estado replicado por Raft em 3-5 nós, via serviço pronto (etcd/Consul) em vez de implementação própria; a queda do líder dispara eleição automática e os workers apenas redirecionam publicações/adoções, sem perder a melhor solução já encontrada. Os heartbeats atuais se generalizam para os heartbeats líder-seguidores do próprio Raft.

### 3.4 Por que a coordenação fica em Python (e não no core C++)

1. **Lei de Amdahl [6]:** a troca ocorre a cada N avaliações e custa microssegundos a milissegundos; a fração sincronizada do tempo total é desprezível. O ganho real de C++ está no caminho quente (decoder), que é ortogonal à cooperação e pode acelerar cada worker individualmente.
2. **Custo de engenharia:** o core C++ é um binário standalone com protocolo textual; memória central em C++ exigiria shared memory POSIX + semáforos ou sockets, duplicando em duas linguagens uma lógica que ainda é objeto de pesquisa. O risco de divergência é real: além do atraso histórico do `min_distance`, a auditoria de out/2026 achou o decoder C++ escolhendo outra caixa, ignorando rotações permitidas, peso e apoio, e um SA nativo sem calibração nem multi-start. A paridade foi restaurada (decoder bit a bit idêntico, `tests/unit/test_native_parity.py`), mas mantê-la tem custo a cada mudança.
3. **Velocidade de iteração científica:** as políticas de cooperação (o que publicar, quando adotar, como preservar diversidade) são exatamente o que se quer variar nos experimentos.

Caminho híbrido registrado como trabalho futuro: coordenador Python + workers do core C++ com ponto de sincronização no protocolo textual.

## 4. Implicações Metodológicas

O modo cooperativo é **não-determinístico**: a ordem das publicações depende do escalonador do sistema operacional. Comparações devem ser feitas por distribuição (médias, medianas, quantis, testes não-paramétricos sobre repetições), nunca seed-a-seed [1]. O modo `process_pool` permanece disponível como referência reprodutível.

Riscos conhecidos e mitigações: colapso prematuro de diversidade (mitigar com intervalo de sincronização como fator experimental e adoção condicionada à distância entre soluções, ex. Kendall-tau); contenção no lock (granularidade grossa já mitiga; medir tempo sob lock); worker travado (heartbeat + deadline, já implementados).

## 5. Perguntas de Pesquisa Abertas

1. A cooperação melhora a qualidade *anytime* (best-so-far por avaliação) frente ao multi-start independente com o mesmo orçamento?
2. Qual o intervalo de sincronização ótimo (trade-off convergência × diversidade)?
3. A adoção deve ser incondicional, probabilística ou guiada por distância entre soluções?
4. Como o ganho escala com workers e tamanho de instância?

## Referências

1. Crainic, T. G. *Parallel Metaheuristics and Cooperative Search.* In: Handbook of Metaheuristics, 3ª ed., Springer, 2019.
2. Crainic, T. G.; Toulouse, M. *Parallel Meta-heuristics.* In: Handbook of Metaheuristics, 2ª ed., Springer, 2010.
3. Lamport, L. *Time, Clocks, and the Ordering of Events in a Distributed System.* Communications of the ACM, 21(7), 1978.
4. Garcia-Molina, H. *Elections in a Distributed Computing System.* IEEE Transactions on Computers, C-31(1), 1982.
5. Ongaro, D.; Ousterhout, J. *In Search of an Understandable Consensus Algorithm (Raft).* USENIX ATC, 2014.
6. Amdahl, G. M. *Validity of the single processor approach to achieving large scale computing capabilities.* AFIPS, 1967.
7. Ropke, S.; Pisinger, D. *An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows.* Transportation Science, 40(4), 2006.
8. Lamport, L. *The Part-Time Parliament (Paxos).* ACM Transactions on Computer Systems, 16(2), 1998.
