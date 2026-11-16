# Estado técnico e científico — 14/11/2026

## Parecer

O projeto tem material para uma apresentação pública como **plataforma experimental de otimização combinatória**: solver legível, core nativo, interface 3D, instrumentação e infraestrutura de experimentos. A abertura permite revisão e reprodução. O anúncio deve descrever essas capacidades, com as limitações atuais.

A contribuição científica ainda está em avaliação. Integrar Extreme Points, SA e otimização bayesiana não basta para estabelecer novidade ou superioridade. Uma demonstração visual também não substitui comparação experimental.

## Qual versão está sendo analisada

Base: `bd34f8f`, na branch `main`; pacote Python **0.3.0**. Os últimos commits de implementação são de 21/10/2026:

| Revisão | Mudança |
|---|---|
| `ef96585` | Fundação 0.2.0: energia, orçamento, restrições e protocolo corrigidos |
| `b52a009` | Reutilização do ajuste do GP contextual por histórico |
| `940d085` | Decoder 0.3.0: first fit entre caixas e projeção de pontos extremos |
| `bd34f8f` | Registro do piloto anterior e da mudança de decoder |

Na inspeção inicial havia uma correção local no registro do piloto e duas notas bibliográficas não versionadas. Todos esses arquivos foram preservados no arquivo privado antes da reorganização.

## Modelo e alcance

É 3D-BPP **offline**, com itens ortogonais conhecidos antes da busca e caixas idênticas. O objetivo principal é reduzir o número de caixas. O SA explora permutações e rotações; o decoder constrói os placements.

No código atual, a energia é `E = b + r`, com `r = 1 − média(u²)` e `u` a ocupação de cada caixa. Para soluções completas com volume positivo, `0 ≤ r < 1`; portanto, uma caixa a menos domina o desempate. Com `b` fixo, o critério favorece concentrar o espaço livre. Esse critério não mede toda a fragmentação geométrica do espaço.

A versão de otimização é NP-difícil. O solver é heurístico e não resolve o problema em geral com certificado de ótimo. A igualdade entre uma solução válida e um limite inferior válido permite certificar somente a contagem de caixas daquela instância. Um gap positivo até um bound frouxo não informa o verdadeiro gap ao ótimo.

## Evidências presentes

- `tests/unit/test_decoder.py` verifica first fit, projeção e três ordenações EP-FFD em 60 instâncias MPV de 50 itens. Os totais de regressão são 833, 822 e 833 caixas; a comparação publicada aceita diferença de até 0,25 caixa na média por classe. Não é reprodução exata de todas as variantes do artigo.
- `tests/unit/test_native_parity.py` compara placements, energia e restrições do decoder Python/C++. As sete entradas sintéticas históricas agora estão em `tests/fixtures/native_parity/`, sem dependência dos resultados arquivados.
- Há testes de orçamento, restrições, separação do dataset, aprendizado, estatística, API e configuração de algoritmos.
- As campanhas registram revisão, ambiente e estado da árvore. Execuções por avaliações são a base da comparação; timeout precisa ser reportado como interrupção de orçamento.

Os números antigos do piloto MPV e do manuscrito não são resultados da 0.3.0. O piloto 0.2.0 não encontrou evidência suficiente nas comparações primárias após Holm; tampouco sustenta superioridade do EGO contextual. A mudança do decoder exige novas campanhas.

## O que está implementado e o que falta demonstrar

| Capacidade | Implementação | Evidência ainda necessária |
|---|---|---|
| SA + Extreme Points | Python e C++ | Qualidade e robustez em campanha atual |
| Multi-start independente | Processos/subprocessos | Speedup e eficiência com orçamento fixo |
| Cooperação e parallel tempering | Python | Benefício sobre baselines com orçamento equivalente |
| GPR + EGO contextual | Surrogate e configuração | Ganho fora da amostra, replicado e com incerteza |
| Restrições físicas simplificadas | Peso total e apoio de base | Centro de massa, resistência e validação física |
| Operador de vizinhança aprendido | Proposta de pesquisa | Implementação e ablação |

O GP exato tem custo cúbico no número de observações; campanhas maiores exigem avaliar agregação, subamostragem ou modelos aproximados. A curadoria sintética seleciona classes com base no comportamento do default no treino e precisa ser declarada como viés de distribuição. O MPV usa orientação fixa e não faz essa curadoria.

## Próximos experimentos

1. Fixar revisão e ambiente; rodar novamente o piloto MPV com 0.3.0 e um único core para todos os braços.
2. Repetir seeds e estudar sensibilidade ao orçamento antes de escolher a contribuição central.
3. Pré-especificar contrastes, unidade experimental `(n, v)` no MPV, família de Holm e dimensão amostral. Seeds não substituem instâncias independentes.
4. Comparar caixas por classe/tamanho com condições de rotação e orçamento explícitas. Não comparar gaps calculados com bounds diferentes como se fossem a mesma métrica.
5. Publicar artefatos compactos e metadados versionados da nova campanha; reservar arquivos grandes para um depósito próprio.

## Comunicação pública

Pode-se afirmar que a plataforma implementa busca, visualização, restrições e configuração experimental. Deve-se evitar anunciar “resolve NP”, “ótimo garantido”, “IA que sempre melhora”, ganho percentual dos estudos antigos ou estado da arte.

O post e o vídeo preparados descrevem funcionamento. A pesquisa sobre operadores aprendidos permanece trabalho futuro. Fontes e conceitos estão em [Referências](references.md), [Benchmark MPV](benchmark_mpv.md) e [Protocolo](experiment_protocol.md).

## Verificação desta preparação

- Snapshot sem histórico Git nem campanhas pessoais: **224 testes passaram**, nenhum skip, em 14/11/2026, com Python 3.14.3 e `ADAPTIVE_3DBP_REQUIRE_NATIVE=1`. A execução levou aproximadamente 75 s. O compilador C++ estava disponível e os testes de paridade foram executados.
- Um aviso de depreciação da integração Starlette/httpx permaneceu; não houve falha da suíte.
- O frontend compilou após instalação limpa e novamente após correções compatíveis de dependências (Vite 7.3.8). A auditoria npm final reportou zero avisos de vulnerabilidade conhecidos. A CI adicionada ainda não rodou no GitHub.
- A interface foi exercitada por captura automatizada com Chrome; a gravação final usa API e solver reais, `persist=false`, 8 buscas em pool com 4 workers C++ e seed 123. O paralelismo foi confirmado por amostragem dos subprocessos e seus tempos de CPU. Resultado e configuração estão em `docs/assets/demo/demo-run.json`.

## Observação de instrumentação

Na primeira captura, com três starts, a série `best_objective_scalar` subiu na transição entre buscas. O histórico concatena melhores locais por start; não é uma curva do incumbente global monotônico. Isso aparece em `SimulatedAnnealingSolver.solve` e no loop multi-start de `cpp/sa_core.cpp`. A solução final é escolhida separadamente pelo melhor score. A captura final usa o pool C++: esse caminho retorna a curva da busca vencedora, com um start por subprocesso, e soma as avaliações dos oito subprocessos no resultado. Corrigir ou rotular o histórico multi-start é uma pendência de instrumentação, sem mudança do solver nesta preparação.

## Distribuição independente — 16/11/2026

Esta árvore inicia um histórico Git próprio a partir dos fontes revisados da versão 0.3.0. Em 16/11/2026, a validação no novo diretório passou: 224 testes com core nativo obrigatório (Python 3.14.3; 1 aviso de depreciação; aproximadamente 72 s), instalação limpa e build do frontend, links locais e seleção de arquivos. Nenhum código de solver, teste, fonte do frontend ou exemplo de entrada foi alterado na transferência. A CI no GitHub é uma verificação adicional, em Python 3.12/3.13 e Node 22.
