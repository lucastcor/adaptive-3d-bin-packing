# Benchmark MPV

O módulo `src/adaptive_3dbp/benchmarks/mpv.py` implementa em Python o procedimento de geração das instâncias associadas a Martello, Pisinger e Vigo. O arquivo público contém a implementação do projeto; o código C original e os PDFs consultados não são distribuídos.

## Convenções implementadas

- Classes 1–8 e tamanhos de 50, 100, 150 e 200 itens.
- Orientação **fixa** por padrão para comparação com os resultados clássicos sem rotação.
- Gerador LCG de 48 bits compatível com `lrand48`, inicializado por `v + n`.
- Ordem dos sorteios e divisão inteira preservadas.
- Classes 1–5 e 8 usam caixas de lado 100; classe 6, lado 10; classe 7, lado 40.
- Treino: índices `v=11..20`; validação: `21..25`; teste canônico: `1..10`.

Classes podem compartilhar itens para o mesmo `(n, v)`. A separação deve ocorrer por índice e sem colisão de `v+n` entre splits de tamanhos diferentes. A estatística pareada usa a unidade `(n, v)`, agregando as classes relacionadas. O gerador de dataset verifica colisões de sementes.

## Uso

```python
from adaptive_3dbp.benchmarks.mpv import mpv_instance

instance = mpv_instance(instance_class=5, n=50, index=1)
```

A spec `configs/campaigns/mpv_pilot.json` gera o dataset e configura a campanha. Isso inicia milhares de execuções; não é necessário para testar a aplicação nem para reproduzir a demonstração.

Os testes `test_mpv_benchmark.py` fixam saídas do RNG e do gerador. `test_decoder.py` compara três ordenações EP-FFD às médias de referência com tolerância por classe. O levantamento histórico registrou comparação de 320 instâncias com OR-Datasets; essa comparação completa não foi refeita nesta preparação pública.

## Fontes

- Martello, Pisinger e Vigo (2000), [The Three-Dimensional Bin Packing Problem](https://doi.org/10.1287/opre.48.2.256.12386).
- Crainic, Perboli e Tadei, [preprint CIRRELT-2007-41](https://www.cirrelt.ca/documentstravail/cirrelt-2007-41.pdf), base das três colunas EP-FFD nos testes.
- Pisinger, [página dos códigos originais](http://hjemmesider.diku.dk/~pisinger/codes.html). O código original declara uso para pesquisa e fins acadêmicos; não se deve atribuir a ele a licença deste projeto.
- [OR-Datasets — Martello2000](https://github.com/Oscar-Oliveira/OR-Datasets/tree/master/Cutting-and-Packing/3D/Datasets/Martello2000), fonte comparativa do levantamento histórico.

Os limites inferiores e os tempos de outros trabalhos não são diretamente intercambiáveis com os deste solver. Para um experimento, reporte orientação, classes, tamanho, índice, seed do solver, avaliações, timeout, core e versão.
