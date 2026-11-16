import { BookOpen } from "lucide-react";
import type { ReactNode } from "react";

type GuideSection = {
  title: string;
  body: ReactNode;
};

const sections: GuideSection[] = [
  {
    title: "★ O pipeline da nossa pesquisa (comece por aqui)",
    body: (
      <>
        <p>
          A ideia central desta plataforma cabe em uma frase: <strong>cada vez que o solver resolve um problema, ele
          aprende algo que melhora a próxima resolução</strong>. Em vez de jogar fora o resultado de cada execução,
          nós o guardamos e usamos como experiência.
        </p>
        <img className="guide-figure" src="/guia/fig1_pipeline.png" alt="Pipeline: instância → solver → solução → banco → GPR → EGO → configuração sugerida" />
        <p>O ciclo tem 4 passos, sempre nesta ordem:</p>
        <ol>
          <li>
            <strong>Resolver.</strong> O Simulated Annealing (SA) empacota os itens. Ele tem ~10 "botões de ajuste"
            (resfriamento, operadores, multi-start…) — e a regulagem ideal desses botões muda de problema para
            problema.
          </li>
          <li>
            <strong>Registrar.</strong> Cada execução vira uma linha no banco: características do problema (quantos
            itens, quão cheios, formatos) + configuração usada + resultado obtido.
          </li>
          <li>
            <strong>Aprender (GPR).</strong> Um modelo estatístico — o Processo Gaussiano — lê esse histórico e aprende
            a prever: "para um problema com esta cara, esta configuração deve render esta qualidade, com esta
            incerteza". A incerteza é o diferencial: o modelo sabe dizer "não sei" onde faltam dados.
          </li>
          <li>
            <strong>Sugerir (EGO).</strong> Na próxima instância, o EGO usa as previsões do GPR para escolher a
            configuração mais promissora — é o botão <em>"Sugerir configuração (EGO)"</em> da aba Solver.
          </li>
        </ol>
        <p>
          <strong>O que nos diferencia das abordagens clássicas?</strong> Três adaptações, cada uma resolvendo um
          problema real:
        </p>
        <ul>
          <li>
            <strong>vs. GPR "puro":</strong> um GPR comum aprende uma função qualquer. O nosso aprende com{" "}
            <em>contexto</em>: as características da instância entram no modelo. Resultado: a sugestão é{" "}
            <em>personalizada por problema</em> (problemas com itens alongados podem ganhar uma configuração diferente
            de problemas com itens cúbicos) — em vez de uma única "configuração média" para tudo.
          </li>
          <li>
            <strong>vs. EGO puro (Jones et al., 1998):</strong> o EGO clássico assume que cada medição é exata. Mas
            rodar o SA duas vezes com seeds diferentes dá resultados diferentes (ruído!). O EGO puro acaba "premiando"
            execuções sortudas — o efeito <em>winner's curse</em>. Nós corrigimos isso de duas formas: modelamos o
            ruído explicitamente (replicações por seed) e usamos o <em>incumbente plug-in</em>: o "melhor até agora" é
            lido da média do modelo, não da medição sortuda.
          </li>
          <li>
            <strong>vs. SMAC3/irace:</strong> esses configuradores buscam <em>uma</em> boa configuração para um
            conjunto de problemas, e o conhecimento morre quando a busca acaba. O nosso sistema é <em>online</em>: o
            conhecimento fica no banco, cresce com o uso e produz sugestões específicas para cada nova instância. Os
            dois entram na campanha justamente como adversários a vencer.
          </li>
        </ul>
        <img className="guide-figure" src="/guia/fig5_incumbente.png" alt="Comparação: incumbente min(y) viesado vs plug-in pela média posterior" />
        <p>
          A figura acima mostra o problema do ruído na prática: os pontos pretos são medições; o ponto circulado em
          vermelho é uma medição "sortuda" bem abaixo da realidade. O EGO puro a usaria como referência (e travaria); a
          linha verde mostra a referência correta, lida da média do modelo. A campanha (aba Experimentos) inclui{" "}
          <em>ablations</em> que desligam cada adaptação para medir quanto cada uma contribui.
        </p>
      </>
    )
  },
  {
    title: "O que esta aplicação faz",
    body: (
      <>
        <p>
          Ela resolve o <strong>problema de empacotamento 3D</strong>: dado um conjunto de itens retangulares e caixas
          iguais, encontrar como posicionar todos os itens usando o <strong>menor número possível de caixas</strong>,
          sem sobreposição. O motor de busca é um <strong>Simulated Annealing</strong> (SA): ele testa milhares de
          ordens e rotações de itens, e um decodificador geométrico (Extreme Points) transforma cada tentativa em
          posições 3D válidas.
        </p>
        <p>
          O resultado nunca usa menos caixas que o <strong>limite inferior</strong> (mínimo teórico provado). Quando o
          solver atinge esse limite, a solução é <strong>ótima comprovada</strong>.
        </p>
      </>
    )
  },
  {
    title: "Instância: JSON e planilha",
    body: (
      <>
        <p>
          A instância descreve a caixa, os itens (dimensões, rotações permitidas) e restrições (ex.: distância mínima
          entre dois itens). Você pode editar o JSON diretamente ou trabalhar com planilha:
        </p>
        <ul>
          <li>
            <strong>Modelo</strong> — baixa uma planilha .xlsx com abas prontas (<code>bin</code>, <code>items</code>,
            <code>min_distance</code>) e instruções na aba LEIA-ME.
          </li>
          <li>
            <strong>Importar</strong> — lê a planilha preenchida e converte para o JSON do solver.
          </li>
          <li>
            <strong>Exportar</strong> — salva a instância atual como planilha editável; após resolver, o botão na
            coluna direita exporta também a <strong>solução</strong> (item, caixa, coordenadas, rotação).
          </li>
        </ul>
      </>
    )
  },
  {
    title: "Parâmetros principais do solver",
    body: (
      <ul>
        <li>
          <strong>Seed</strong> — semente da aleatoriedade. Mesma seed + mesma configuração = mesma trajetória
          (nos modos determinísticos). Use seeds diferentes para medir robustez.
        </li>
        <li>
          <strong>Avaliações</strong> — orçamento de soluções candidatas testadas. Mais avaliações = mais chance de
          melhorar, mais tempo.
        </li>
        <li>
          <strong>Tempo</strong> — limite de parede em segundos; o que acabar primeiro (tempo ou avaliações) encerra.
        </li>
        <li>
          <strong>Resfriamento e Iter./T</strong> — controlam a “temperatura” do SA: quente aceita pioras para explorar;
          frio refina. Resfriamento perto de 1.0 esfria devagar.
        </li>
        <li>
          <strong>Multi-start</strong> — <em>número de buscas independentes</em>: o solver dispara N partidas, cada uma
          começando de uma solução inicial diferente (maior volume primeiro, maior dimensão, aleatórias…), e fica com a
          melhor. É o seguro contra começar a busca num caminho ruim — e é o que os modos paralelos distribuem entre os
          workers.
        </li>
        <li>
          <strong>Amostra hist.</strong> — de quantas em quantas avaliações o histórico é registrado (1 = todos os
          pontos; valores maiores deixam a resposta mais leve).
        </li>
      </ul>
    )
  },
  {
    title: "Core Python vs C++",
    body: (
      <p>
        O <strong>Python</strong> é o core de referência: mais fácil de auditar e suporta tudo. O <strong>C++</strong>{" "}
        executa o mesmo algoritmo compilado, reduzindo o overhead por avaliação — útil para orçamentos grandes. Se o
        binário não estiver disponível na sua máquina, a aplicação cai automaticamente para o Python e avisa nos
        diagnósticos.
      </p>
    )
  },
  {
    title: "Paralelismo: Sequencial, Pool e Cooperativo",
    body: (
      <>
        <p>Pense nas partidas do multi-start como exploradores procurando o melhor caminho:</p>
        <ul>
          <li>
            <strong>Sequencial</strong> — um explorador por vez. Mais lento, porém 100% reprodutível e com o melhor
            diagnóstico. Suporta pausa.
          </li>
          <li>
            <strong>Pool independente</strong> — vários exploradores simultâneos que <em>não conversam</em>; no final,
            vence o melhor. Reprodutível seed a seed. Como os processos não têm canal de comunicação, <em>não suporta
            pausa</em> (interromper descarta ao final).
          </li>
          <li>
            <strong>Pool cooperativo</strong> — exploradores que compartilham o melhor mapa encontrado a cada
            “sincronização” (intervalo configurável). Costuma convergir mais rápido, mas a ordem das trocas depende do
            sistema operacional: <em>não-determinístico</em> — avalie por média de seeds. Suporta pausa (o sinal viaja
            pela mesma memória central).
          </li>
        </ul>
        <p>
          <strong>Quantos workers?</strong> O solver é limitado por CPU; a GPU não é usada. A regra prática: até o
          número de <em>núcleos físicos</em>, deixando 1 livre para o sistema. A aba <strong>Sistema</strong> mostra a
          recomendação calculada para a sua máquina (SO, arquitetura ARM/x86, núcleos e memória) e o consumo em tempo
          real durante a execução.
        </p>
      </>
    )
  },
  {
    title: "Executar, pausar, retomar e interromper",
    body: (
      <>
        <p>
          A execução roda como um job em segundo plano: a barra de progresso mostra % (por avaliações ou tempo), e os
          gráficos da Análise atualizam ao vivo.
        </p>
        <ul>
          <li>
            <strong>Pausar/Retomar</strong> — disponível no Sequencial, no Cooperativo e no core C++ (em macOS/Linux).
            A pausa congela a busca e o relógio do orçamento.
          </li>
          <li>
            <strong>Interromper</strong> — cancela o job; o resultado parcial é descartado e nada é gravado no banco.
          </li>
        </ul>
      </>
    )
  },
  {
    title: "Lendo os resultados",
    body: (
      <ul>
        <li>
          <strong>Visualização 3D</strong> — cada caixa em uma aba; arraste para orbitar, role para zoom.
        </li>
        <li>
          <strong>Caixas vs limite inferior</strong> — se “caixas = limite”, a solução é ótima comprovada; senão, o{" "}
          <strong>gap</strong> mostra a distância máxima possível do ótimo.
        </li>
        <li>
          <strong>Curva de convergência</strong> — “Melhor” deve descer em degraus; “Atual” oscila (o SA aceita pioras
          de propósito); “Aceite” começa alto e cai com o resfriamento; “Temp.” mostra o resfriamento em si.
        </li>
        <li>
          <strong>Utilização por caixa</strong> — quanto do volume de cada caixa foi aproveitado.
        </li>
      </ul>
    )
  },
  {
    title: "Aprendizado (GPR/EGO) e runs salvas",
    body: (
      <p>
        Com o SQLite ligado, cada execução é salva e alimenta um modelo <strong>GPR</strong> que aprende “qual
        configuração funciona para qual tipo de instância”. Esse conhecimento é a base do <strong>EGO</strong>, que
        sugere configurações promissoras. As runs salvas aparecem na coluna direita — clique para recarregar instância,
        configuração e resultado. Nos experimentos repetidos, deixe a persistência <em>desligada</em> para não enviesar
        o aprendizado.
      </p>
    )
  },
  {
    title: "Campanha: dataset, estratégias e parâmetros",
    body: (
      <>
        <p>
          A campanha da aba <strong>Experimentos</strong> não roda sobre a instância aberta na aba Solver. Ela compara
          estratégias de configuração do Simulated Annealing em um <strong>dataset com treino, validação e teste</strong>.
          Por padrão esse dataset é gerado automaticamente; se você tiver um <code>manifest.json</code> salvo, pode colar
          o caminho dele no modo <strong>Manifest local</strong>.
        </p>
        <ul>
          <li>
            <strong>Estratégia/configurador</strong> — método que escolhe parâmetros do SA. A configuração fixa não
            aprende; Random Search sorteia; EGO usa um GP; SMAC3 e irace são configuradores externos.
          </li>
          <li>
            <strong>Classes</strong> — famílias sintéticas de instâncias. Cada classe combina tamanho, ocupação alvo,
            forma dos itens, heterogeneidade e, opcionalmente, restrição <code>min_distance</code>.
          </li>
          <li>
            <strong>Train/classe</strong> — quantas instâncias de treino são criadas para cada classe. Nesta tela,
            validação e teste recebem 1 instância por classe no dataset gerado.
          </li>
          <li>
            <strong>Trials</strong> — quantas configurações candidatas cada estratégia pode testar no treino. Depois,
            a melhor configuração é reavaliada em validação e teste.
          </li>
          <li>
            <strong>Seeds</strong> — repetições aleatórias do solver. O treino usa as seeds informadas; validação usa
            <code> seed+10000</code>; teste usa <code>seed+20000</code>, iguais para todos os braços.
          </li>
          <li>
            <strong>Aval./run</strong> e <strong>Tempo/run</strong> — orçamento de uma execução individual do solver:
            1 instância × 1 seed × 1 configuração. A campanha inteira soma muitas dessas execuções.
          </li>
        </ul>
        <p>
          O resultado comparável é a <strong>perda em validação</strong>; o teste deve ser lido uma única vez. Os
          downloads incluem <code>runs.csv</code>, <code>report.md</code>, <code>metadata.json</code> e o manifest do
          dataset usado, para repetir ou auditar a campanha.
        </p>
      </>
    )
  },
  {
    title: "Aba Experimentos: comparando variantes",
    body: (
      <>
        <p>
          O laboratório roda a <strong>mesma instância</strong> com as variantes selecionadas (Sequencial, Pool,
          Cooperativo) e as <strong>mesmas seeds</strong>, em sequência, e agrega os resultados:
        </p>
        <ul>
          <li>
            <strong>Caixas (média/melhor)</strong> — qualidade da solução por variante.
          </li>
          <li>
            <strong>Tempo médio e Aval./s</strong> — custo e vazão.
          </li>
          <li>
            <strong>Speedup</strong> — tempo do Sequencial ÷ tempo da variante (quanto o paralelismo acelerou{" "}
            <em>nesta máquina</em>).
          </li>
        </ul>
        <p>
          Use ao menos 3 seeds; para o Cooperativo, a média entre seeds é a única comparação honesta, pois cada execução
          pode trocar soluções em ordens diferentes.
        </p>
      </>
    )
  }
];

export function GuidePanel() {
  return (
    <div className="guide-page">
      <section className="panel guide-panel" aria-label="Guia da interface">
        <div className="panel-heading">
          <BookOpen size={18} aria-hidden="true" />
          <h2>Guia rápido</h2>
        </div>
        <p className="guide-intro">
          Manual didático da interface: o que cada controle faz, como interpretar os gráficos e como tirar proveito do
          paralelismo na sua máquina. Documentação científica completa em <code>docs/</code> no repositório.
        </p>
        {sections.map((section, index) => (
          <details className="guide-section" key={section.title} open={index === 0}>
            <summary>{section.title}</summary>
            <div className="guide-body">{section.body}</div>
          </details>
        ))}
      </section>
    </div>
  );
}
