# Dane ewaluacyjne

Ten plik opisuje, jak korzystać z materiałów ewaluacyjnych. Zawartość zbioru i jego pomiary określa
`manifest.json`, parametry przeprowadzonych pomiarów zapisano w `protocol.json`, a zasady oceny
przebiegów względem klucza znajdują się w `scoring-protocol.json`.

Wszystkie oceny powstają za pomocą `scripts/evaluate.py`. Sędzia otrzymuje całe zapisane przebiegi,
ale ocenia osobno każde ustalenie merytoryczne względem klucza i tekstów źródeł, zgodnie
z `scoring-protocol.json`. To jedyna procedura punktowania; nie towarzyszą jej osobne pliki
dopasowania ustaleń ani niezależne tabele jakości.

## Zawartość katalogu

| Ścieżka | Zawartość |
| --- | --- |
| `manifest.json` | Dane każdego dokumentu: identyfikator, pochodzenie (syntetyczny lub rzeczywisty), przypadek badawczy, oznaczenie wersji podstawowej, ścieżka, skróty, przydział do części zbioru, prawa, status prywatności i wyniki pomiarów odczytu |
| `source-files/acquisition.json` | Adres źródła, czas pobrania i dwa zestawy skrótów dla każdego pliku: `acquired_*` dla wersji udostępnionej przez wydawcę, `stored_*` dla przechowywanej tutaj kopii po usunięciu danych osobowych z metadanych |
| `protocol.json` | Parametry pomiaru, podział zbioru, warunki wykonania i zachowane zapisy przebiegów |
| `scoring-protocol.json` | Konfiguracja sędziego, kryteria oceny, miary jakości i ich mianowniki, progi porównań oraz stan gotowości do oceny |
| `answer-key.json` | Oczekiwane ustalenia, dokładne fragmenty dokumentów, dopuszczalne pary kodu i podstawy prawnej oraz uzasadnienia oparte na źródłach |
| `documents/` | Dokumenty syntetyczne przygotowane na potrzeby projektu |
| `source-files/` | Dokumenty rzeczywiste po usunięciu danych osobowych z metadanych |
| `results/` | Surowe wyniki prób rozwojowych i pomiarów końcowych; nazwa katalogu opisuje ich pochodzenie, ale sama nie rozstrzyga, czy dany przebieg należy do ewaluacji |
| `judge/sources/` | Teksty dokumentów i przepisów używane do oceny, ze skrótami treści i potwierdzoną zgodnością wersji korpusu |
| `judge/control/` | Przypadki rozwojowe i kontrolne, oceny referencyjne oraz ustalone kryteria akceptacji sędziego |
| `results/judge/calibration-verdicts.jsonl` | Zapisane odpowiedzi sędziego dla przypadków rozwojowych i kontrolnych |
| `results/parity.json` | Zestawienie porównań konfiguracji w obu seriach asystenta |
| `results/judge/control-report.json` | Porównanie zapisanych odpowiedzi kontrolnych z ocenami referencyjnymi, obliczane offline |
| `results/judge/verdicts.jsonl` | Zapisane próby i zweryfikowane oceny ustaleń asystenta |
| `results/judge/report.json` | Miary jakości zapisane jako ułamki, oceny poszczególnych ustaleń, porównania, liczby błędów, zużycie zasobów i skróty danych wejściowych |
| `LICENSE.md` | Oświadczenie CC0 dotyczące plików syntetycznych i własnych metadanych projektu oraz informacje o pochodzeniu dwóch plików pobranych od wydawcy |

Dokładne wartości należy odczytywać z `manifest.json`. Odwołuj się do tego pliku zamiast tworzyć
osobną kopię jego danych w opisie.

Dwie zachowane serie mają w `scoring-protocol.json` numery 1 i 2. Ich pola `batch` wskazują
niezmienione katalogi wejściowe.

## Stan ewaluacji

Dwie zarejestrowane serie zawierają 18 przebiegów i 410 ustaleń. Zatwierdzony klucz pozostaje
niezmienny podczas oceny. Sędzia otrzymuje 18 pakietów obejmujących całe przebiegi i 301 ustaleń
merytorycznych; pozostałe 109 ustaleń liczy się osobno. Każdy pakiet zawiera pełny dokument i teksty
źródeł bez powtarzania ich przy kolejnych ustaleniach. Sędzia musi podać osobną ocenę z identyfikatorem
dla każdego ustalenia merytorycznego.

Kontrola sędziego na podstawie źródeł i pełna ocena zostały zakończone. Kontrola spełniła kryteria
akceptacji: 10/12 ocen końcowych było zgodnych z ocenami referencyjnymi przygotowanymi z pomocą modelu,
bez rozbieżności w dopasowaniu do klucza i bez błędnych zapewnień o zgodności. Raport jakości obejmuje
wszystkie 410 ustaleń, w tym 301 zweryfikowanych ocen z 18 pakietów całych przebiegów. Obok poprawnych
odpowiedzi zachowano jedną odrzuconą próbę z niepoprawnym cytatem i uwzględniono jej zużycie zasobów.
Poniższe polecenia pozwalają odtworzyć raport kontroli i raport jakości bez połączenia z modelem.

## Podział zbioru

Podział ustalono dla każdego przypadku badawczego, więc wszystkie jego wersje w różnych formatach
należą do tej samej części zbioru. Skrypt `scripts/validate_evaluation_data.py` zawiera ustaloną
tabelę podziału i zgłasza błąd przy każdej rozbieżności z manifestem. Celowo zapisano ją osobno:
kontrola oparta wyłącznie na sprawdzanym pliku nie wykryłaby zmiany podziału w tym pliku.

Dokumenty testowe pozostają niewykorzystane aż do pomiaru końcowego, z jednym odnotowanym wyjątkiem:
ED-001 posłużył do próby porównawczej 2026-09-04, przed pomiarem końcowym. Zapisano to w `protocol.json`
(`holdout_discipline.ed_001_exposure`), a surowe wyniki znajdują się w
`results/development/2026-09-04-ed-001-holdout-exposure/`. Ograniczenia wynikające z tej próby oraz
zasady korzystania z części rozwojowej określa `protocol.json` w sekcji `holdout_discipline`.

## Oddzielenie danych od procesu analizy

Do ocenianego wariantu wolno przekazać wyłącznie treść wybranego dokumentu. Ten README,
`manifest.json`, `acquisition.json`, `protocol.json`, `scoring-protocol.json`, nazwy plików i etykiety
typów są metadanymi badawczymi, które ujawniają sposób przygotowania zbioru. Każdy wariant musi
otrzymać identyczną treść, co do bajtu. Klucz służy wyłącznie do oceny wyników i nigdy nie trafia
na wejście wariantu: podanie oczekiwanego ustalenia pozwalałoby je skopiować zamiast samodzielnie
rozpoznać.

## Odtwarzanie wyników i kontrole

Wszystkie poniższe polecenia uruchamiaj z głównego katalogu artefaktu. Zainstaluj zależności
w ustalonych wersjach poleceniem `uv sync --locked --all-groups`.

### Odtworzenie zapisanych statystyk bez modelu

```bash
uv run python scripts/check_judge.py \
  --cases evaluation-data/judge/control/cases.json \
  --reference evaluation-data/judge/control/reference-ratings.json \
  --criteria evaluation-data/judge/control/acceptance.json \
  --verdicts evaluation-data/results/judge/calibration-verdicts.jsonl \
  --protocol evaluation-data/scoring-protocol.json \
  --output evaluation-data/results/judge/control-report.json

uv run python scripts/evaluate.py report --offline
```

Polecenia odczytują zapisane odpowiedzi i źródła. Nie wymagają uruchomionego backendu, połączenia
z modelem ani danych dostępowych. Podczas tworzenia raportu program sprawdza skróty danych
wejściowych, ponownie weryfikuje odpowiedzi i zapisuje status każdego ustalenia asystenta.
Niepełny zestaw ocen daje niepełny raport i niezerowy kod wyjścia. Identyczne dane wejściowe dają
identyczny JSON. Każda miara jakości zachowuje licznik i mianownik; wartość, której nie da się
obliczyć, jest zapisana jako `null`.

### Odtworzenie kontroli zgodności konfiguracji

Pliki surowych odpowiedzi zawierają konfigurację każdego przebiegu. Poniższa kontrola odczytuje je
tym samym mechanizmem sprawdzającym skróty, którego używa program oceny. Porównuje wszystkie
zarejestrowane pary wewnątrz serii oraz odpowiadające sobie przebiegi między seriami, a liczby
porównań sprawdza względem `results/parity.json`. Nie wymaga połączenia z bazą ani modelem.

```bash
PYTHONPATH=scripts uv run python - <<'PYTHON'
import json
from pathlib import Path
from types import SimpleNamespace
from contract_analyzer.agents.parity import compare_arm_parity
from evaluation_inputs import ARMS, load_json, load_runs

root = Path('evaluation-data')
protocol = load_json(root / 'scoring-protocol.json')
runs = load_runs(root, protocol).cells
series = sorted({cell[0] for cell in runs})
documents = protocol['scored_runs']['cohort']
assert len(runs) == len(series) * protocol['scored_runs']['cells_per_series']
assert len({run['id'] for run in runs.values()}) == len(runs)
assert all(run['parent_run_id'] is None for run in runs.values())
summary = {kind: {'checked': 0, 'failed': 0}
           for kind in ('within_series', 'between_series')}

def check(kind, left, right):
    result = compare_arm_parity([SimpleNamespace(**runs[left]), SimpleNamespace(**runs[right])])
    summary[kind]['checked'] += 1
    summary[kind]['failed'] += int(not result.measurement_valid)

comparisons = load_json(root / 'protocol.json')['registered_comparisons']
for number in series:
    for document in documents:
        for comparison in comparisons:
            left, right = [arm.lower() for arm in comparison['pair']]
            check('within_series', (number, document, left), (number, document, right))
for number in series[1:]:
    for document in documents:
        for arm in ARMS:
            check('between_series', (series[0], document, arm), (number, document, arm))
assert summary == load_json(root / 'results/parity.json')['summary']
assert all(value['failed'] == 0 for value in summary.values())
print(json.dumps(summary, indent=2))
PYTHON
```

### Uruchomienie oceny przez model

Sędzią jest GPT-5.6 Sol z poziomem rozumowania `high`, wywoływany przez bibliotekę `openai-codex`
w ustalonej wersji, z użyciem subskrypcji ChatGPT. Zaloguj się przez zewnętrzny program Codex CLI
poleceniem `codex login`; `codex login status` sprawdza stan logowania. Domyślnym katalogiem danych
dostępowych jest `~/.codex`. Aby wskazać inny, użyj `--credentials-home`. Sędzia wymaga
uwierzytelnienia kontem ChatGPT i nie zastępuje go kluczem API.

```bash
# Sprawdź przygotowanie pakietów bez wywoływania modelu
uv run python scripts/evaluate.py packets

# Oceń ustalone przypadki kontrolne, a następnie odtwórz i sprawdź raport kontroli opisany powyżej
uv run python scripts/evaluate.py calibrate \
  --cases evaluation-data/judge/control/cases.json --workers 4 --credentials-home ~/.codex

# Oceń brakujące przebiegi po zaliczeniu kontroli i potwierdzeniu gotowości do oceny
uv run python scripts/evaluate.py run --workers 4 --credentials-home ~/.codex

# Przelicz statystyki z zapisanych ocen
uv run python scripts/evaluate.py report --offline
```

Polecenie `run` wymaga sprawdzonego klucza i protokołu gotowego do oceny. Wykonuje najwyżej cztery
niezależne oceny równolegle. Każde zadanie ma oddzielne środowisko wykonania, a każdy przebieg
otrzymuje nowy kontekst oceny. Każda próba oceny całego przebiegu jest zapisywana od razu. Po przerwie
technicznej użyj tego samego polecenia, aby kontynuować z tym samym modelem, danymi wejściowymi
i konfiguracją. Program wykorzystuje ponownie kompletną ocenę, która przeszła sprawdzenie, niezależnie
od jej wniosków. Brakujące, powtórzone lub obce identyfikatory ustaleń unieważniają odpowiedź.
Czas i tokeny sędziego są raportowane oddzielnie od zużycia zasobów asystenta i liczone jeden raz
dla każdej próby oceny przebiegu.

Przygotowane źródła znajdują się w zestawie. Aby odtworzyć je z zadeklarowanego korpusu, uruchom:

```bash
uv run python scripts/evaluate.py prepare --corpus-manifest corpus/manifest.example.json
```

Przygotowanie wymaga dostępu do wskazanych źródeł prawnych i sprawdza zgodność wersji korpusu
z zapisanymi przebiegami. Sędzia otrzymuje te teksty na wejściu i nie pobiera dodatkowych źródeł.
Przygotowanie źródeł ani ocenianie nie uruchamiają ponownie asystenta.

### Kontrola spójności zbioru

Uruchom z głównego katalogu artefaktu:

```bash
uv run python scripts/validate_evaluation_data.py
```

```bash
uv run python scripts/measure_evaluation_ingestion.py --tokenizer <path-to>/tokenizer.json --check
```

Używaj `uv run python` lub `.venv/bin/python` zamiast samego `python`. Jeśli `python` wskazuje
na inny interpreter, drugie polecenie kończy się błędem `ModuleNotFoundError: No module named
'tokenizers'`. Oznacza to brak zależności w używanym środowisku, a nie problem z danymi.

Pierwszy skrypt korzysta wyłącznie z biblioteki standardowej. Sprawdza zgodność manifestu z plikami,
identyfikatory, skróty i liczebność dokumentów syntetycznych oraz zgodność dokumentów rzeczywistych
z zapisem pobrania. Kontroluje też dopuszczalne wartości statusów praw i prywatności, ustalony podział
zbioru, wersję podstawową każdego przypadku oraz obecność, wartości liczbowe i spójność opisu
długości dokumentów. **Nie potwierdza** prawdziwości `rights_status`; podstawą są zapisane obok
dowody źródłowe. Dlatego przytoczono je dosłownie zamiast streszczać.

Drugi skrypt ponownie odczytuje każdy dokument, dzieli go na jednostki i rozpoznaje odwołania
mechanizmami aplikacji. Zgłasza błąd, jeśli którykolwiek wynik różni się od zapisanego pomiaru
**lub pomiaru nie można wykonać na danej maszynie**. Pomyślne zakończenie oznacza zatem sprawdzenie
wszystkich pięciu dokumentów. Skrypt wymaga zależności aplikacji; konwersja dwóch plików `.doc`
wymaga LibreOffice, dostępnego w obrazie kontenera. Żaden dokument w zestawie nie jest skanem,
więc OCR nie jest uruchamiany. Skrypt nie wywołuje modelu.

## Prawa i usuwanie metadanych osobowych

Dwa wzory ministerialne są udostępniane na warunkach ponownego wykorzystywania określonych przez
wydawcę. `LICENSE.md` zawiera wymagane przy przekazywaniu kopii informacje o źródle, autorstwie
i redakcji, znaczniki czasu plików oraz opis przetworzenia. `manifest.json` zapisuje zweryfikowany
status praw, a `source-files/acquisition.json` pierwotne metadane i wyniki sprawdzenia skrótów.

Z dokumentów przekazywanych systemowi usunięto metadane zawierające imiona i nazwiska. Informacje
o pochodzeniu zachowano osobno w nocie licencyjnej i zapisie pobrania.

## Ograniczenia

Zbiór dobrano celowo, aby objąć wybrane zagadnienia. Nie odzwierciedla ich częstości występowania
i nie pozwala wnioskować o wszystkich polskich umowach. Dokumenty służą do badań; **nie są wzorami
umów do wykorzystania ani poradą prawną**.
