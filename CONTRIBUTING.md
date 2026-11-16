# Contribuir

Relatos reproduzíveis de bugs, melhorias de documentação, novas instâncias de teste e comparações com orçamento controlado são bem-vindos. Confira a situação da licença em `docs/public-release.md` antes de redistribuir código.

## Desenvolvimento

Siga a instalação do README e execute:

```bash
./scripts/build_native.sh
ADAPTIVE_3DBP_REQUIRE_NATIVE=1 python -m pytest
cd frontend
npm ci
npm run build
```

Mudanças no decoder, energia ou restrições precisam preservar a validação geométrica e atualizar os testes de paridade Python/C++. Mudanças no protocolo devem explicitar efeitos sobre a comparabilidade de resultados antigos.

Em relatos de execução, inclua versão do pacote, revisão Git, versão do Python, core, configuração, seed, instância pequena e diagnóstico. Remova caminhos pessoais e dados privados dos anexos.

## Experimentos

Leia `docs/experiment_protocol.md`. Fixe avaliações, declare timeouts e separe treino/validação/teste. Não apresente seeds como réplicas independentes das instâncias. Não misture campanhas de versões com comportamentos diferentes.

## Arquivos

Versione fontes, testes, fixtures pequenas e documentação. Mantenha artigos de terceiros como referências e links. Bancos pessoais, modelos treinados, notas privadas e saídas geradas pertencem a áreas ignoradas. Mídias escolhidas para o README ficam em `docs/assets/demo/`.
