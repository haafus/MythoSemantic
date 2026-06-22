# Индукция мотивов в тексте: методы и алгоритмы (обзор для исследователя)

## TL;DR

- «Индукция мотивов в тексте» — это не одна задача, а семейство из шести во многом независимых традиций: (1) нарратологические/фольклорные мотивы (Thompson Motif-Index, ATU, computational folkloristics), (2) повторяющиеся текстовые/последовательные паттерны (суффиксные структуры, n-граммы, sequential pattern mining, text reuse), (3) тематическое моделирование (LDA и наследники), (4) эмбеддинговые/нейронные методы (BERTopic, контекстные эмбеддинги), (5) перенос алгоритмов motif discovery из time-series/биоинформатики (matrix profile, SAX), (6) сетевые мотивы в текстовых графах (FANMOD, mfinder; нарративные/персонажные сети).
- Текущий передний край (2023–2026) — это LLM-методы: лучшая опубликованная система детекции нарративных мотивов — fine-tuned Llama-3 (~0.85 F1 на «Arabian Nights», препринт 2026), а индуктивный тематический анализ через промптинг LLM конкурирует с ручным кодированием; параллельно классические primitives (matrix profile, суффиксные деревья, ассоциативные меры) остаются стандартом для поверхностных и символьных паттернов.
- Главная нерешённая проблема — оценка: нет общепринятых бенчмарков и метрик для «семантических»/нарративных мотивов, аннотирование дорого и культурно-зависимо, а LLM склонны к «фантомным» примерам и нестабильности; рекомендуется комбинировать поверхностные методы (для воспроизводимости) с LLM (для семантической глубины) и обязательной валидацией человеком-носителем культуры.

## Key Findings

1. **Шесть смыслов «мотива».** Термин «motif» в анализе текста скрывает радикально разные объекты: от точной повторяющейся подстроки до «наименьшего элемента сказки, способного сохраняться в традиции» (определение Стита Томпсона). Методология определяется тем, какой смысл подразумевается.
1. **Фольклорные мотивы оставались ручными до ~2010-х.** Thompson Motif-Index (TMI, 1955–1958) содержит около 46 000 мотивов и подмотивов (по Uther 2004 «over 46,000 motifs remain applicable to the more than 250 newly added or revised types»; сам Томпсон классифицировал «nearly 50,000 motifs»); ATU (Uther 2004) — комплементарный индекс типов сказок. Автоматизация началась с призывов Darányi (2010) и работ Declerck & Lendvai (2011), Finlayson и Yarlott.
1. **Классический pattern mining зрелый и точный.** Суффиксные деревья (Ukkonen 1995), sequential pattern mining (GSP, SPADE, PrefixSpan), ассоциативные меры для коллокаций — дают поверхностные, точные, воспроизводимые мотивы, но не «понимают» семантику.
1. **Topic modeling = индукция тем как распределений слов.** LDA (Blei, Ng, Jordan 2003) — фундамент; нейронный наследник BERTopic (Grootendorst 2022) даёт более когерентные темы на коротких/многоязычных текстах.
1. **Перенос из time-series.** Matrix Profile (Yeh et al. 2016, серия UCR) и SAX (Lin, Keogh et al. 2003) позволяют находить мотивы в символьных последовательностях; SAX прямо создавался, чтобы «использовать мощь алгоритмов обработки текста» для рядов, и наоборот применим к токенам.
1. **Сетевые мотивы.** mfinder (Milo et al. 2002), FANMOD/RAND-ESU (Wernicke & Rasche 2006) — для подграфов; применяются к со-встречаемостным и персонажным сетям (Elson et al. 2010; обзор Labatut & Bost 2019).
1. **LLM-эра.** Narrative event chains (Chambers & Jurafsky 2008/2009) → event schema induction промптингом (Li et al. 2023) → индуктивный тематический анализ LLM (De Paoli 2024) и детекция мотивов fine-tuned LLM (Alyami & Finlayson 2026).

## Details

### 1. Нарративные/фольклорные мотивы (computational folkloristics)

**Концептуальная основа.** Стит Томпсон определял мотив как «наименьший элемент сказки, обладающий способностью сохраняться в традиции» (Thompson 1977: 415). TMI (Thompson, *Motif-Index of Folk-Literature*, 6 томов, 1955–1958) классифицирует мотивы в 23 темы по буквам (A: Mythology, B: Animals, C: Tabu, D: Magic…) и делит их на события, персонажей и предметы; индекс ссылается на >614 коллекций и содержит около 46 248 мотивов и подмотивов. ATU (Aarne–Thompson–Uther, Uther 2004) — комплементарный индекс типов сказок; номера ATU идут от 1 до 2499 с тремя главными разделами: 1–299 (Animal Tales), 300–1199 (Ordinary Folktales) и 1200–2499 (Jokes, Anecdotes and Formula Tales). Критика (Dundes 1997): перекрывающиеся подкатегории, цензура, евроцентризм.

**Цифровые ресурсы и ранняя автоматизация.** MOMFER (Karsdorp, van der Meulen, Meder, van den Bosch 2015, *Folklore* 126(1):37–52) — поисковик по TMI с WordNet-обогащением. Declerck & Lendvai (2011) и Declerck et al. (2012) конвертировали TMI/ATU в формат для многоязычной индексации (но по описаниям мотивов, без привязки к текстам). Darányi (2010; Darányi & Forró 2012) предложил идею мотивов как «нарративной ДНК» — последовательностей; Ofek, Darányi & Rokach (2013, CMN) показали обучение типов сказок по последовательностям мотивов. Dutch Folktale Database (Meertens Instituut, основана в 1994) — крупнейший фольклорный архив: «The Folktale Database of the Low Countries contains more than 100,000 folktales»; в 2024 она была дополнена сказками из Flemish Folktale Database, а исходно индексировала >42 000 голландских сказок по ATU+TMI.

**Школа Финлейсона (структура и мотивы).** Mark Finlayson создал Story Workbench (2008) для глубокой аннотации и алгоритм **Analogical Story Merging (ASM)** (Finlayson 2011/2012, диссертация MIT *Learning narrative structure from annotated folktales*) на основе байесовского слияния моделей (Bayesian model merging) — он выучил существенную часть морфологии Проппа из 15 размеченных русских сказок, достигнув Rand Index 0.511 относительно функций Проппа. Это «первая демонстрация вычислительной системы, выучивающей реальную теорию нарративной структуры».

**Детекция мотивов в современном тексте (Yarlott & Finlayson).** Yarlott & Finlayson (2016, CMN, *Learning a Better Motif Index*) — формальное определение мотива и архитектура системы. «Finding Trolls Under Bridges» (Yarlott, Ochoa, Acharya, … Finlayson 2022, arXiv:2204.06085) — прототип детектора; off-the-shelf детектор метафор как признак дал F1=0.35 на мотивах. Ключевой ресурс — **GOLEM** (Yarlott, Acharya, Castro Estrada, Gomez, Finlayson 2024, LREC-COLING 2024, с. 7801–7813): корпус из 7 955 английских статей (2 039 424 слова), 26 078 кандидатов в мотивы по 34 типам из трёх культурных групп (еврейская, ирландская, пуэрто-риканская), размеченных по 4 классам употребления (Motific, Referential, Eponymic, Unrelated), давших 1 723 реально мотивных употребления; Fleiss κ > 0.55. Классификация типа употребления — трудная задача: few-shot на T5, FLAN-T5, GPT-2, Llama 2 (7B) дал лучшую точность лишь 41%. **MIME** (Acharya, Estrada, Dahal, Yarlott, Gomez, Finlayson 2024, NLP+CSS workshop, с. 46–56, *Discovering Implicit Meanings of Cultural Motifs from Text*) — proof-of-concept, извлекающий имплицитный смысл мотива из индексов, Wikipedia, информантов и соцсетей; вывод — явная информация от носителей критична для качества.

**SOTA по детекции мотивов: «Arabian Nights».** Alyami & Finlayson (2026, препринт arXiv:2603.19283, *Automated Motif Indexing on the Arabian Nights*) — «первый вычислительный подход к индексации мотивов». Использован индекс El-Shamy (2006) «A Motif Index of The Thousand and One Nights» (~5 000 мотивов, по схеме TMI), привязанный к тексту (выравнивание изданий Burton↔Irwin алгоритмом Needleman-Wunsch + WordNet-синонимы, точность 0.99). Корпус: 2 670 выражений мотивов 200 различных мотивов в 58 450 предложениях; Cohen κ = 0.72. Сравнено 5 подходов: (1) retrieve-and-rerank (BM25 + all-mpnet-base-v2 + BERT cross-encoder); (2) off-the-shelf эмбеддинги (mistral-embed, Gemini text-embedding-004, NV-Embed-v2, jina-embeddings-v3, sentence-t5-base); (3) дообученные эмбеддинги (SBERT, sentence-t5); (4) генеративный промптинг (Mistral-7B, Llama-3.1-8B) zero/few-shot; (5) те же с LoRA. **Лучший результат — fine-tuned Llama-3: 0.85 F1** (0.86 P / 0.84 R); Mistral-FT 0.81; few-shot Llama3 0.77. На мотивах простой структуры — 0.90 F1, на сложных выражениях — 0.73 F1. Retrieve-and-rerank худший (0.36 F1). *Кавеат: это непрорецензированный препринт.*

**LLM для типов сказок.** Arčon, Robnik-Šikonja, Tratnik (2025, arXiv:2510.18561, кейс Cinderella) — zero-shot GPT-4.5 Preview детектирует наличие/отсутствие мотивов в вариантах «Золушки»; матрицы присутствия мотивов кластеризуются; эмбеддинги LaBSE + HDBSCAN для семантической близости; кросс-язычный (включая словенский) анализ. Также Gervás & Méndez и работы по тэггингу функций Проппа через LLM (CEUR Vol-3671).

**Школа Тангерлини (макроскоп фольклора).** Timothy Tangherlini (UCLA/Berkeley) — «computational folkloristics» (спецвыпуск *Journal of American Folklore* 129(511), 2016). WitchHunter (Broadwell & Tangherlini 2016) — гео-семантический браузер по >30 000 датских сказок; Abello, Broadwell & Tangherlini (2012, *Communications of the ACM*, *Computational Folkloristics*) — сети из ~2 973 узлов / 52 663 рёбер для поиска похожих историй.

### 2. Повторяющиеся текстовые/последовательные паттерны

**Суффиксные структуры.** Суффиксное дерево (алгоритм Укконена 1995, линейное время O(n)) находит самую длинную повторяющуюся подстроку, наиболее частые подстроки длины ≥ k за линейное/почти линейное время; суффиксные массивы (Manber & Myers) и FM-index (Ferragina & Manzini, на основе Burrows-Wheeler) — компактные альтернативы. Гранулярность — точная поверхностная подстрока. VizTree (Lin, Keogh) — визуализация паттернов через суффиксные деревья.

**Sequential pattern mining.** Задача — частые упорядоченные подпоследовательности. GSP (Srikant & Agrawal 1996, Apriori-подобный), SPADE (Zaki 2000, вертикальный формат, пересечение id-списков), PrefixSpan (Pei, Han et al. 2001/2004, IEEE TKDE, projection-based pattern growth; обычно быстрее GSP/FreeSpan/SPADE), а также SPAM, BIDE, CloSpan (закрытые паттерны), CM-SPADE. Применяются к токенным последовательностям для извлечения формульных/синтаксических паттернов.

**Коллокации и ассоциативные меры.** Извлечение мультисловных выражений через PMI, log-likelihood ratio, chi-square, t-score, Dice (обзоры Evert 2004/2009; Pecina 2010). Manning & Schütze (*Foundations of Statistical NLP*) — каноническая ссылка. Extension patterns (Petrović et al.) обобщают биграммные меры на n-граммы.

**Text reuse.** Обнаружение повторно используемых фрагментов в исторических корпусах: TRACER/eTRAP (Büchler et al. 2014), **Passim** (Smith, Cordell et al. 2014, на основе локального выравнивания), применение **BLAST** из биоинформатики (Vesanto et al. 2017 — финские газеты; Vierthaler & Gelein 2019 — китайские корпуса). ReceptionReader (Helsinki, 2024/2025) — пайплайн на ~миллиард BLAST-хитов по корпусам ECCO/EEBO-TCP. Вывод нескольких работ (Manjavacas et al. 2019): эмбеддинги пока не дают существенного преимущества над классическим IR для аллюзивного reuse.

### 3. Тематическое моделирование как индукция тем

**LDA** (David M. Blei, Andrew Y. Ng, Michael I. Jordan, «Latent Dirichlet Allocation», *Journal of Machine Learning Research* 3(Jan):993–1022, 2003 — «a generative probabilistic model for collections of discrete data such as text corpora») — порождающая модель: документ = смесь тем, тема = распределение над словами; вывод через вариационный байес или коллапсированное сэмплирование Гиббса (Griffiths & Steyvers 2004). Расширения: Correlated Topic Models и Dynamic Topic Models (Blei & Lafferty 2006/2007), supervised LDA (McAuliffe & Blei 2008), online LDA (Hoffman et al. 2010). Предшественники: LSI (Deerwester et al. 1990), pLSI (Hofmann 1999). Гранулярность — мягкие тематические кластеры на уровне корпуса; порядок слов игнорируется (bag-of-words).

### 4. Эмбеддинговые/нейронные методы

**BERTopic** (Grootendorst 2022, arXiv:2203.05794) — эмбеддинги (SBERT) → понижение размерности (UMAP) → плотностная кластеризация (HDBSCAN) → c-TF-IDF для представления темы. Превосходит LDA/NMF по когерентности на коротких/многоязычных/нишевых текстах. Родственники: Top2Vec, Contextualized Topic Models (CTM). Семантическая индукция мотивов как кластеризация эмбеддингов предложений/абзацев. Используется как этап в LLM-пайплайнах извлечения тем (с последующей LLM-меткой).

### 5. Перенос алгоритмов motif discovery из time-series

**SAX** (Lin, Keogh, Lonardi, Chiu 2003) — символьное представление рядов: z-нормализация → PAA (кусочно-агрегированная аппроксимация, Keogh et al. 2001) → дискретизация в алфавит по равновероятным интервалам N(0,1); нижне-ограничивающая метрика расстояния. Прямо создан, чтобы применять текстовые алгоритмы (хэширование, суффиксные деревья) к рядам — и обратимо применим к токенным последовательностям. EMMA/HOT-SAX, SAX-VSM — варианты.

**Matrix Profile** (Yeh, Zhu, Ulanova … Mueen, Keogh 2016, ICDM, *Matrix Profile I*) — для каждой подпоследовательности хранит расстояние до ближайшего соседа (z-нормализованное евклидово); top-k мотивы = минимумы профиля. Серия UCR (Matrix Profile I–XXVII): STAMP, STOMP, SCRIMP++ (интерактивная скорость), VALMOD (переменная длина), motif discovery под DTW (Alaee, Kamgar, Keogh 2020). Унифицирует motif/discord/shapelet discovery. Применимо к символьным последовательностям текста.

### 6. Сетевые мотивы в текстовых графах

**Network motifs** (Milo et al. 2002, *Science*) — статистически переусиленные подграфы относительно случайной нулевой модели (z-score). Инструменты: **mfinder** (Kashtan et al. 2002, перечисление + сэмплирование, до размера 7), **FANMOD** (Wernicke & Rasche 2006, алгоритм RAND-ESU, на порядки быстрее, поддержка цветных графов), MAVisto, Kavosh, G-Tries, MODA, NetMODE. Сетевой vs мотив-центричный подход.

**Текстовые/нарративные сети.** Персонажные сети: Elson, Dames & McKeown (2010, ACL, на основе диалоговых признаков); обзор Labatut & Bost (2019, *ACM Computing Surveys* 52(5):89) — трёхфазный фреймворк (идентификация персонажей через NER+alias resolution → детекция взаимодействий → построение графа); пайплайны Renard, Charnetto. Narrative Smoothing (Bost et al. 2016) для динамических сетей сериалов. «Maths Meets Myths» (network analysis древних нарративов). Над такими графами можно искать сетевые мотивы (триады, мотивы взаимодействия).

### 7. Связующие нейронные/LLM-подходы

**Narrative event chains / schema induction.** Chambers & Jurafsky (2008, ACL, *Unsupervised Learning of Narrative Event Chains*; 2009 — schemas + participants) — частично упорядоченные множества событий вокруг общего протагониста; обучение через дистрибутивные меры + темпоральный классификатор; оценка через narrative cloze. Развитие: graph-based schema induction (Li et al. 2021, arXiv:2104.06344), incremental prompting and verification на LLM (Li, Zhao, Li, Ji, Callison-Burch, Han 2023, arXiv:2307.01972), human-in-the-loop schema induction (2023).

**Индуктивный тематический анализ через LLM.** De Paoli (2024, *Social Science Computer Review* 42(4):997–1019) — «an experiment done with the LLM GPT 3.5-Turbo to perform an inductive Thematic Analysis (TA)» на полуструктурированных интервью. Deiner et al. (2024, *JMIR Infodemiology*) — LLM проводят индуктивный тематический анализ корпуса соцсетей в одном промпте с валидацией человеком (отмечены «фантомные» примеры). Множество работ 2024–2025 адаптируют 6 фаз Braun & Clarke (reflexive thematic analysis) к LLM; LOGOS (2025) — end-to-end grounded theory; обзоры по prompt engineering для LLM-ITA (Khalid & Witmer 2025). Экономические/новостные нарративы: RELATIO (Ash et al. 2024, на основе semantic role labeling), извлечение экономических нарративов интегрированным LLM-подходом (2025, arXiv:2506.15041).

### Оценка, бенчмарки, датасеты

- **Фольклор:** TMI, ATU, Dutch Folktale Database, аннотированный корпус Финлейсона из 15 русских сказок (18 862 слова), Annotated Folktales (aft) corpus, GOLEM (26 078 кандидатов), El-Shamy index + Arabian Nights (2 670 выражений). Метрики: F1/precision/recall, Cohen/Fleiss κ для согласия аннотаторов, Rand Index (для кластеризации структуры).
- **Topic modeling:** topic coherence (C_v, NPMI), topic diversity, перплексия; датасеты 20 Newsgroups, корпус Science.
- **Time-series:** UCR Time Series Classification Archive (Dau, Keogh et al. 2018); недавняя работа «Time Series Motif Discovery: A Comprehensive Evaluation» (VLDB 2025).
- **Event schemas:** narrative cloze, order coherence.
- **Сетевые мотивы:** z-score значимости относительно ансамбля случайных графов (E. coli, дрожжи, C. elegans — каноничные тест-сети).
- **LLM-thematic analysis:** нет золотого стандарта; используют согласие с человеческим кодированием, Cohen κ + семантическое сходство, inductive thematic saturation (De Paoli & Mathis 2025).

### Ключевые площадки, группы, авторы

- **Площадки:** ACL/EMNLP/NAACL/LREC-COLING, Workshop on Computational Models of Narrative (CMN, Dagstuhl OASIcs), *Journal of Cultural Analytics*, *Digital Scholarship in the Humanities*, *Journal of American Folklore*, *Journal of Open Humanities Data*, KDD/ICDM/VLDB/SIGMOD (pattern & time-series mining), *Bioinformatics*/PLOS (сетевые мотивы), CHR (Computational Humanities Research).
- **Группы/авторы:** Mark Finlayson и W. Victor Yarlott (FIU — детекция мотивов); Timothy Tangherlini, Peter Broadwell, James Abello (computational folkloristics); Folgert Karsdorp, Theo Meder, Antal van den Bosch, Thierry Declerck (фольклорные индексы/NLP); David Blei (topic models); Eamonn Keogh, Jessica Lin, Abdullah Mueen (time-series motifs); Nathanael Chambers & Dan Jurafsky (narrative schemas); Vincent Labatut & Xavier Bost (персонажные сети); Maarten Grootendorst (BERTopic); Sebastian Wernicke, Ron Milo (сетевые мотивы); Jiawei Han, Mohammed Zaki (sequential pattern mining).

## Recommendations

1. **Сначала определите смысл «мотива».** Если нужны точные повторяющиеся фразы/формулы — начните с суффиксных деревьев/массивов и ассоциативных мер (быстро, воспроизводимо, без обучения). Если нужны темы — LDA как бейзлайн, BERTopic как современный дефолт. Если нужны нарративные/культурные мотивы — переходите к LLM-пайплайнам.
1. **Для нарративных мотивов в 2026:** воспроизведите пайплайн Alyami & Finlayson (fine-tuned Llama-3 / Mistral с LoRA) — это текущий SOTA (~0.85 F1), но требует размеченного корпуса и индекса мотивов. Для эксплоративного анализа без разметки — zero/few-shot промптинг (GPT-4-класс) + кластеризация матриц присутствия мотивов (подход Arčon et al.).
1. **Для тематической/мотивной индукции на больших корпусах:** двухэтапно — BERTopic/эмбеддинговая кластеризация для генерации кандидатов → LLM для именования и интерпретации тем. Это снимает проблему нечитаемых тем LDA и галлюцинаций «с нуля».
1. **Для символьных/последовательных представлений:** SAX + Matrix Profile, если текст превращается в числовой/символьный ряд (напр., эмоциональные арки, частотные сигналы); используйте VALMOD для мотивов переменной длины.
1. **Для структурных мотивов взаимодействий:** стройте персонажные сети (пайплайн Labatut & Bost / Renard) и применяйте FANMOD для значимых подграфов.
1. **Всегда валидируйте человеком-носителем** для культурных мотивов и используйте κ-согласие; для LLM-тем измеряйте inductive thematic saturation и проверяйте «фантомные» примеры.
1. **Пороги смены стратегии:** если F1 поверхностных методов < ~0.5 на семантических мотивах (как в GOLEM, 41% accuracy у few-shot) — переходите к fine-tuning; если когерентность тем LDA низкая на коротких текстах — переходите на BERTopic; если объём корпуса делает попарные сравнения невозможными — используйте Matrix Profile (GPU) или BLAST-хэширование.

## Caveats

- Поле фрагментировано: шесть традиций почти не цитируют друг друга, единой терминологии и бенчмарка нет.
- Часть ключевых результатов — препринты, не прошедшие рецензирование (в частности, Arabian Nights 0.85 F1, arXiv:2603.19283, дата март 2026): цифры заявлены авторами.
- LLM-методы нестабильны (зависимость от промпта, «фантомные» цитаты, дрейф версий моделей) и плохо воспроизводимы; для строгого научного вывода поверхностные методы предпочтительнее.
- Оценка семантических/нарративных мотивов остаётся открытой проблемой: дорогое культурно-зависимое аннотирование, низкое межаннотаторское согласие на тонких различиях, риск евроцентризма индексов (TMI/ATU критикуются Dundes 1997).
- Перенос алгоритмов из time-series/биоинформатики к тексту требует осторожной дискретизации; «мотив» в смысле matrix profile (числовая подпоследовательность) семантически не эквивалентен фольклорному мотиву.