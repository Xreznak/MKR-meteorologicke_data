# ==============================================================
# MODUL: Generovanie HTML reportu zo všetkých analýz
# ==============================================================

import os            # Práca so súborovým systémom (cesty, kontrola existencie súborov)
import io            # In-memory textový buffer pre zachytávanie výstupu
import base64        # Kódovanie obrázkov priamo do HTML (self-contained súbor)
import contextlib    # Zachytávanie výstupu funkcií (redirect_stdout)
import pandas as pd  # Načítanie CSV so štatistikami a tvorba HTML tabuliek

# Naše analytické moduly
from nacitanie_dat     import nacitaj_data
from analyza_chyb      import (analyza_chybajucich_dni, analyza_chybajucich_vzoriek,
                                detekcia_poskodennych_vzoriek, histogramy_dlzok_chyb)
from analyza_ziarenia  import rocna_analyza_ziarenia, denna_analyza_pre_mesiace
from analyza_korelacie import korelacie_parametrov, suhrne_statistiky


# Fyzikálne medze – rovnaké ako v analyza_chyb.py
FYZIKALNE_MEDZE = {
    'Irradiance':               (0,    1500),
    'BodyTemperature':          (-40,  80),
    'RelativeHumidity':         (0,    100),
    'HumidityTemp':             (-40,  60),
    'Pressure':                 (800,  1100),
    'PressureAvg':              (800,  1100),
    'TiltAngle':                (-90,  90),
    'FanSpeed':                 (0,    15000),
    'SunZenith':                (0,    180),
    'SunAzimuth':               (0,    360),
}

# Stĺpce, ktoré zobrazíme v tabuľke chybných vzoriek
ZOBRAZIT_STLPCE = ['DateTime', 'Irradiance', 'BodyTemperature', 'RelativeHumidity',
                   'Pressure', 'PressureAvg', 'SunZenith', 'SunAzimuth',
                   'TiltAngle', 'FanSpeed']


# ==============================================================
# POMOCNÉ FUNKCIE
# ==============================================================

def zachyt_vystup(funkcia, *args, **kwargs):
    """
    Spustí funkciu a zachytí všetok jej textový výstup (print) do reťazca.
    Vracia: (návratová_hodnota, zachytený_text)
    """
    buffer = io.StringIO()                          # Dočasný textový buffer
    with contextlib.redirect_stdout(buffer):        # Presmerujeme stdout do buffra
        vysledok = funkcia(*args, **kwargs)         # Spustíme funkciu
    return vysledok, buffer.getvalue()              # Vrátime výsledok + zachytený text


def tabulka_poskodennych(df):
    """
    Nájde všetky poškodené vzorky (mimo fyzikálnych medzi) a vráti HTML tabuľku.
    Chybné hodnoty sú červeno zvýraznené, ostatné bunky sú normálne.
    """
    # Nájdeme masku poškodených riadkov a pre každý stĺpec masku mimo medzi
    riadky_maska  = pd.Series(False, index=df.index)   # Celková maska poškodených riadkov (OR všetkých stĺpcov)
    stlpce_chyby  = {s: pd.Series(False, index=df.index) for s in FYZIKALNE_MEDZE}   # Maska chýb pre každý stĺpec zvlášť

    for stlpec, (mn, mx) in FYZIKALNE_MEDZE.items():   # Pre každý sledovaný stĺpec s jeho medzami
        if stlpec not in df.columns:   # Preskočíme stĺpce, ktoré v dátach nie sú
            continue
        mimo = ((df[stlpec] < mn) | (df[stlpec] > mx)) & df[stlpec].notna()   # Hodnoty mimo medzi (NaN ignorujeme)
        stlpce_chyby[stlpec] = mimo    # Uložíme masku chýb pre tento stĺpec
        riadky_maska = riadky_maska | mimo   # Pridáme do celkovej masky (riadok je chybný ak má aspoň jednu chybu)

    # Vyberieme len poškodené riadky a len zobrazované stĺpce
    dostupne = [s for s in ZOBRAZIT_STLPCE if s in df.columns]   # Stĺpce, ktoré skutočne existujú v dátach
    df_chyby = df[riadky_maska][dostupne].copy()   # Vyfiltrujeme len chybné riadky

    if df_chyby.empty:
        return '<p style="color:#2e7d32; font-weight:600;">✓ Žiadne poškodené vzorky nenájdené.</p>'

    # Zostavenie HTML tabuľky ručne – aby sme vedeli zvýrazniť konkrétne bunky
    hlavicka = ''.join(f'<th>{s}</th>' for s in dostupne)   # HTML hlavička tabuľky zo zoznamu stĺpcov
    riadky_html = []   # Zoznam HTML reťazcov pre každý riadok tabuľky

    for idx, row in df_chyby.iterrows():   # Iterujeme cez každý poškodený riadok
        bunky = []   # Zoznam HTML buniek pre aktuálny riadok
        for stlpec in dostupne:   # Iterujeme cez každý zobrazovaný stĺpec
            hodnota = row[stlpec]   # Hodnota tejto bunky
            # Skontrolujeme, či je táto konkrétna bunka mimo medze
            je_chybna = (stlpec in stlpce_chyby) and stlpce_chyby[stlpec].get(idx, False)

            if je_chybna:
                # Červená bunka s hodnotou a povoleným rozsahom v tooltipe
                mn, mx = FYZIKALNE_MEDZE[stlpec]   # Načítame povolené medze pre tento stĺpec
                bunky.append(
                    f'<td class="chybna-bunka" title="Povolený rozsah: [{mn}, {mx}]">'
                    f'{hodnota}</td>'   # Bunka s červeným štýlom a tooltipom
                )
            else:
                # Normálna bunka
                if isinstance(hodnota, float):   # Číselné hodnoty zaokrúhlíme na 4 desatinné miesta
                    bunky.append(f'<td>{hodnota:.4f}</td>')
                else:
                    bunky.append(f'<td>{hodnota}</td>')   # Ostatné hodnoty (napr. dátum) bez formátovania

        riadky_html.append(f'<tr>{"".join(bunky)}</tr>')   # Zostavíme celý riadok tabuľky

    return f"""
    <p style="margin-bottom:12px; color:#555;">
      Zobrazených <strong>{len(df_chyby)}</strong> poškodených vzoriek.
      <span style="color:#c62828;">■</span> Červená bunka = hodnota mimo fyzikálneho rozsahu
      (podržte myš pre zobrazenie povoleného rozsahu).
    </p>
    <div style="overflow-x:auto;">
    <table class="chyby-tabulka">
      <thead><tr>{hlavicka}</tr></thead>
      <tbody>{''.join(riadky_html)}</tbody>
    </table>
    </div>"""


def obrazok_na_base64(cesta):
    """
    Načíta PNG obrázok a zakóduje ho do base64 reťazca.
    Výsledok môžeme vložiť priamo do HTML ako <img src="data:image/png;base64,...">
    Výhodou je, že HTML je samostatný súbor bez externých závislostí.
    """
    if not os.path.exists(cesta):
        return None
    with open(cesta, 'rb') as f:          # Čítame binárne (rb = read binary)
        data = f.read()
    return base64.b64encode(data).decode('utf-8')   # Kódujeme a dekódujeme na string


def text_na_html(text):
    """
    Konvertuje zachytený textový výstup na formátovaný HTML blok.
    Zachováva odsadenie, zvýrazní riadky so štatistikami.
    """
    riadky = []   # Zoznam HTML riadkov výsledného bloku
    for riadok in text.strip().split('\n'):   # Rozdelíme text na jednotlivé riadky
        # Prázdny riadok
        if not riadok.strip():   # Prázdny riadok → vložíme zalomenie riadku
            riadky.append('<br>')
            continue
        # Nadpis sekcie (riadky s ===)
        if riadok.strip().startswith('==='):   # Riadok je nadpis sekcie (obalený ===)
            obsah = riadok.strip().strip('=').strip()   # Odoberieme znaky = a biele znaky
            riadky.append(f'<div class="sekcia-nadpis">{obsah}</div>')   # Nadpis sekcie
        # Oddeľovač (===, ---)
        elif set(riadok.strip()) <= set('=-'):   # Riadok pozostáva len z = a - znakov
            continue   # Preskočíme čistý oddeľovač
        # Riadok s číselnou hodnotou (obsahuje : a číslicu)
        elif ':' in riadok and any(c.isdigit() for c in riadok):   # Riadok obsahuje kľúč: hodnota
            cast = riadok.split(':', 1)   # Rozdelíme na kľúč a hodnotu (len pri prvej dvojbodke)
            riadky.append(
                f'<div class="hodnota-riadok">'
                f'<span class="kluc">{cast[0]}:</span>'   # Kľúč (parameter)
                f'<span class="hodnota">{cast[1]}</span>'  # Hodnota (číslo / reťazec)
                f'</div>'
            )
        else:
            # Obyčajný riadok
            escaped = riadok.replace('<', '&lt;').replace('>', '&gt;')   # Escapujeme HTML znaky
            riadky.append(f'<div class="info-riadok">{escaped}</div>')   # Bežný informačný riadok
    return '\n'.join(riadky)   # Spojíme všetky HTML riadky do jedného reťazca


# ==============================================================
# HLAVNÁ FUNKCIA – zostavenie HTML reportu
# ==============================================================

def generuj_html_report(csv_subor, output_dir):
    """
    Spustí celú analýzu, zachytí výstup a vygeneruje self-contained HTML report.
    Obrázky sú vložené priamo do HTML ako base64 → jeden prenosný súbor.
    """
    print("\n=== GENEROVANIE HTML REPORTU ===")

    # Načítanie dát
    print("  Načítavam dáta...")   # Informujeme o začatí načítavania
    df = nacitaj_data(csv_subor)   # Načítame a pripravíme DataFrame zo CSV súboru

    # ----------------------------------------------------------
    # Spustenie všetkých analýz so zachytením výstupu
    # ----------------------------------------------------------
    print("  Spúšťam analýzy...")   # Informujeme o začatí analýz

    _, txt_statistiky   = zachyt_vystup(suhrne_statistiky,           df, output_dir)   # Súhrnné štatistiky + uloženie CSV
    _, txt_dni          = zachyt_vystup(analyza_chybajucich_dni,      df, output_dir)   # Analýza chýbajúcich dní
    _, txt_nan          = zachyt_vystup(analyza_chybajucich_vzoriek,  df)               # Analýza NaN hodnôt
    (poskodene, pravd), txt_poskodene = zachyt_vystup(                                  # Detekcia hodnôt mimo fyzikálnych medzi
        detekcia_poskodennych_vzoriek, df
    )
    _, txt_dlzky        = zachyt_vystup(histogramy_dlzok_chyb,        df, poskodene, output_dir)   # Histogramy dĺžok skupín chýb
    _, txt_rocna        = zachyt_vystup(rocna_analyza_ziarenia,        df, output_dir)             # Ročná analýza žiarenia
    _, txt_mesiacna     = zachyt_vystup(denna_analyza_pre_mesiace,     df, output_dir)             # Mesačné denné profily
    _, txt_korelacie    = zachyt_vystup(korelacie_parametrov,          df, output_dir)             # Korelačná matica

    # ----------------------------------------------------------
    # Načítanie obrázkov ako base64
    # ----------------------------------------------------------
    print("  Načítavam obrázky...")   # Informujeme o začatí kódovania obrázkov

    def img(nazov):
        """Skratka: načíta obrázok z output_dir a vráti base64 string."""
        return obrazok_na_base64(os.path.join(output_dir, nazov))   # Zostavíme cestu a zakódujeme

    obrazky = {   # Slovník: kľúč → base64 reťazec obrázku (alebo None ak neexistuje)
        'statistiky_csv':         os.path.join(output_dir, 'suhrne_statistiky.csv'),   # CSV súbor štatistík
        'histogram_dni':          img('histogram_chybajucich_dni.png'),                # Histogram chýbajúcich dní
        'histogram_vzoriek':      img('histogram_vzoriek_za_den.png'),                 # Distribúcia vzoriek za deň
        'vzorky_casovy':          img('vzorky_za_den_casovy_priebeh.png'),             # Časový priebeh vzoriek
        'histogram_poskodene':    img('histogram_dlzok_poskodennych.png'),             # Histogram dĺžok chýb
        'rocna_analyza':          img('rocna_analyza_ziarenia.png'),                   # Ročná analýza žiarenia
        'mesiacna_analyza':       img('denna_analyza_mesiacov.png'),                   # Mesačné denné profily
        'heatmapa':               img('heatmapa_korelacie.png'),                       # Korelačná heatmapa
        'scatter':                img('scatter_korelacie.png'),                        # Scatter ploty korelácie
    }

    # Načítanie CSV štatistík do HTML tabuľky
    stats_csv = os.path.join(output_dir, 'suhrne_statistiky.csv')   # Cesta k CSV so štatistikami
    if os.path.exists(stats_csv):   # Skontrolujeme, či súbor existuje
        df_stats = pd.read_csv(stats_csv, index_col=0)               # Načítame štatistiky do DataFrame
        tabulka_html = df_stats.round(3).to_html(classes='stats-tabulka', border=0)   # Konverzia na HTML tabuľku
    else:
        tabulka_html = '<p>Štatistiky nie sú dostupné.</p>'   # Náhradný text ak CSV chýba

    # ----------------------------------------------------------
    # Pomocná funkcia pre vkladanie obrázkov
    # ----------------------------------------------------------
    def img_tag(kluc, alt='', sirka='100%'):
        """Vráti <img> tag s base64 obrázkom, alebo prázdny reťazec ak obrázok chýba."""
        if obrazky.get(kluc):   # Skontrolujeme, či obrázok bol úspešne načítaný
            return (f'<img src="data:image/png;base64,{obrazky[kluc]}" '   # Vložíme base64 dáta priamo do src
                    f'alt="{alt}" style="width:{sirka}; max-width:100%;">')
        return f'<p class="chyba">Obrázok "{alt}" nie je dostupný.</p>'   # Fallback ak obrázok chýba

    # ----------------------------------------------------------
    # Zostavenie HTML
    # ----------------------------------------------------------
    print("  Zostavujem HTML...")   # Informujeme o začatí zostavovania HTML dokumentu

    html = f"""<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Štatistická analýza meteorologických dát</title>
  <style>
    /* ---- Základné štýly ---- */
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f0f2f5;
      color: #222;
      line-height: 1.6;
    }}

    /* ---- Hlavička ---- */
    header {{
      background: linear-gradient(135deg, #1a237e, #283593);
      color: white;
      padding: 40px 60px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    header h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 8px; }}
    header p  {{ font-size: 1rem; opacity: 0.85; }}

    /* ---- Navigácia ---- */
    nav {{
      background: #283593;
      padding: 12px 60px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }}
    nav a {{
      color: #c5cae9;
      text-decoration: none;
      font-size: 0.85rem;
      padding: 4px 10px;
      border-radius: 12px;
      transition: background 0.2s;
    }}
    nav a:hover {{ background: rgba(255,255,255,0.15); color: white; }}

    /* ---- Obsah ---- */
    main {{ max-width: 1300px; margin: 30px auto; padding: 0 30px; }}

    /* ---- Sekcie ---- */
    .sekcia {{
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
      margin-bottom: 30px;
      overflow: hidden;
    }}
    .sekcia-hlavicka {{
      background: #e8eaf6;
      padding: 18px 28px;
      border-left: 5px solid #3949ab;
    }}
    .sekcia-hlavicka h2 {{
      font-size: 1.2rem;
      color: #1a237e;
      font-weight: 600;
    }}
    .sekcia-telo {{ padding: 24px 28px; }}

    /* ---- Karta metriky ---- */
    .karty {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }}
    .karta {{
      background: #f5f5f5;
      border-radius: 10px;
      padding: 16px 22px;
      min-width: 160px;
      flex: 1;
      border-top: 4px solid #3949ab;
    }}
    .karta-nadpis {{ font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
    .karta-hodnota {{ font-size: 1.6rem; font-weight: 700; color: #1a237e; margin-top: 4px; }}
    .karta-jednotka {{ font-size: 0.8rem; color: #888; }}

    /* ---- Výstupný text ---- */
    .vystup-text {{
      background: #f8f9ff;
      border: 1px solid #e0e3ff;
      border-radius: 8px;
      padding: 16px 20px;
      font-family: 'Consolas', 'Courier New', monospace;
      font-size: 0.82rem;
      margin-bottom: 20px;
      overflow-x: auto;
    }}
    .sekcia-nadpis {{
      font-weight: 700;
      color: #283593;
      margin: 10px 0 6px;
      font-size: 0.9rem;
    }}
    .hodnota-riadok {{ display: flex; gap: 8px; padding: 2px 0; }}
    .kluc  {{ color: #555; min-width: 260px; }}
    .hodnota {{ color: #1a237e; font-weight: 600; }}
    .info-riadok {{ color: #444; padding: 1px 0; }}

    /* ---- Tabuľka štatistík ---- */
    .stats-tabulka {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      margin-bottom: 20px;
      overflow-x: auto;
      display: block;
    }}
    .stats-tabulka th {{
      background: #3949ab;
      color: white;
      padding: 10px 14px;
      text-align: right;
      white-space: nowrap;
    }}
    .stats-tabulka th:first-child {{ text-align: left; }}
    .stats-tabulka td {{
      padding: 8px 14px;
      text-align: right;
      border-bottom: 1px solid #eee;
    }}
    .stats-tabulka td:first-child {{ text-align: left; font-weight: 600; color: #3949ab; }}
    .stats-tabulka tr:hover {{ background: #f0f4ff; }}

    /* ---- Obrázky ---- */
    .obrazok-kontainer {{ margin: 16px 0; }}
    .obrazok-kontainer img {{
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      display: block;
    }}
    .obrazok-popis {{
      font-size: 0.8rem;
      color: #888;
      text-align: center;
      margin-top: 6px;
    }}
    .dvojica {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 800px) {{ .dvojica {{ grid-template-columns: 1fr; }} }}

    /* ---- Päta ---- */
    footer {{
      text-align: center;
      padding: 30px;
      color: #888;
      font-size: 0.8rem;
    }}

    .chyba {{ color: #aaa; font-style: italic; font-size: 0.85rem; }}

    /* ---- Tabuľka chybných vzoriek ---- */
    .chyby-tabulka {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
      font-family: 'Consolas', monospace;
    }}
    .chyby-tabulka th {{
      background: #b71c1c;
      color: white;
      padding: 9px 12px;
      text-align: left;
      white-space: nowrap;
      position: sticky;
      top: 0;
    }}
    .chyby-tabulka td {{
      padding: 7px 12px;
      border-bottom: 1px solid #eee;
      white-space: nowrap;
    }}
    .chyby-tabulka tr:hover td {{ background: #fff8f8; }}
    .chybna-bunka {{
      background: #ffebee !important;
      color: #c62828;
      font-weight: 700;
      border: 1px solid #ef9a9a !important;
      cursor: help;
    }}
  </style>
</head>
<body>

<!-- HLAVIČKA -->
<header>
  <h1>Štatistická analýza meteorologických dát</h1>
  <p>Vstupný súbor: <strong>{csv_subor}</strong> &nbsp;|&nbsp; Výstupný adresár: <strong>{output_dir}/</strong></p>
</header>

<!-- NAVIGÁCIA -->
<nav>
  <a href="#statistiky">Štatistiky</a>
  <a href="#chybajuce-dni">Chýbajúce dni</a>
  <a href="#chybajuce-vzorky">Chýbajúce vzorky</a>
  <a href="#poskodene">Poškodené vzorky</a>
  <a href="#rocna">Ročná analýza</a>
  <a href="#mesiacna">Mesačná analýza</a>
  <a href="#korelacie">Korelácie</a>
</nav>

<main>

<!-- ============================================================ -->
<!-- 1. SÚHRNNÉ ŠTATISTIKY -->
<!-- ============================================================ -->
<div class="sekcia" id="statistiky">
  <div class="sekcia-hlavicka"><h2>1. Súhrnné štatistiky parametrov</h2></div>
  <div class="sekcia-telo">
    <div class="karty">
      <div class="karta">
        <div class="karta-nadpis">Celkový počet vzoriek</div>
        <div class="karta-hodnota">{len(df):,}</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Počet dní s dátami</div>
        <div class="karta-hodnota">{df['Date'].nunique()}</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Priemerné žiarenie</div>
        <div class="karta-hodnota">{df['Irradiance'].mean():.1f}</div>
        <div class="karta-jednotka">W/m²</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Maximum žiarenia</div>
        <div class="karta-hodnota">{df['Irradiance'].max():.1f}</div>
        <div class="karta-jednotka">W/m²</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Priemerný tlak</div>
        <div class="karta-hodnota">{df['Pressure'].mean():.0f}</div>
        <div class="karta-jednotka">hPa</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Priemerná teplota</div>
        <div class="karta-hodnota">{df['BodyTemperature'].mean():.1f}</div>
        <div class="karta-jednotka">°C</div>
      </div>
    </div>
    {tabulka_html}
    <div class="vystup-text">{text_na_html(txt_statistiky)}</div>
  </div>
</div>

<!-- ============================================================ -->
<!-- 2. CHÝBAJÚCE DNI -->
<!-- ============================================================ -->
<div class="sekcia" id="chybajuce-dni">
  <div class="sekcia-hlavicka"><h2>2. Analýza chýbajúcich dní a počtu vzoriek</h2></div>
  <div class="sekcia-telo">
    <div class="karty">
      <div class="karta">
        <div class="karta-nadpis">Chýbajúce dni</div>
        <div class="karta-hodnota" style="color:#c62828;">
          {len(set(pd.date_range(df['DateTime'].min().date(), df['DateTime'].max().date(), freq='D').date) - set(df['Date'].unique()))}
        </div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Priem. vzoriek / deň</div>
        <div class="karta-hodnota">{df.groupby('Date').size().mean():.1f}</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Min vzoriek / deň</div>
        <div class="karta-hodnota">{df.groupby('Date').size().min()}</div>
      </div>
      <div class="karta">
        <div class="karta-nadpis">Max vzoriek / deň</div>
        <div class="karta-hodnota">{df.groupby('Date').size().max()}</div>
      </div>
    </div>
    <div class="vystup-text">{text_na_html(txt_dni)}</div>
    <div class="dvojica">
      <div class="obrazok-kontainer">
        {img_tag('histogram_dni', 'Histogram chýbajúcich dní')}
        <div class="obrazok-popis">Histogram chýbajúcich dní podľa mesiaca</div>
      </div>
      <div class="obrazok-kontainer">
        {img_tag('histogram_vzoriek', 'Histogram vzoriek za deň')}
        <div class="obrazok-popis">Distribúcia počtu vzoriek na deň</div>
      </div>
    </div>
    <div class="obrazok-kontainer" style="margin-top:20px;">
      {img_tag('vzorky_casovy', 'Vzorky v čase')}
      <div class="obrazok-popis">Počet vzoriek za každý deň v čase</div>
    </div>
  </div>
</div>

<!-- ============================================================ -->
<!-- 3. CHÝBAJÚCE VZORKY (NaN) -->
<!-- ============================================================ -->
<div class="sekcia" id="chybajuce-vzorky">
  <div class="sekcia-hlavicka"><h2>3. Chýbajúce vzorky (NaN hodnoty)</h2></div>
  <div class="sekcia-telo">
    <div class="vystup-text">{text_na_html(txt_nan)}</div>
  </div>
</div>

<!-- ============================================================ -->
<!-- 4. POŠKODENÉ VZORKY -->
<!-- ============================================================ -->
<div class="sekcia" id="poskodene">
  <div class="sekcia-hlavicka"><h2>4. Poškodené vzorky a dĺžky súvislých skupín chýb</h2></div>
  <div class="sekcia-telo">
    <div class="karty">
      <div class="karta" style="border-top-color:#c62828;">
        <div class="karta-nadpis">Pravdepodobnosť chybnej vzorky</div>
        <div class="karta-hodnota" style="color:#c62828;">{pravd*100:.4f} %</div>
      </div>
    </div>
    <div class="vystup-text">{text_na_html(txt_poskodene + txt_dlzky)}</div>
    <div class="obrazok-kontainer">
      {img_tag('histogram_poskodene', 'Histogram dĺžok poškodených vzoriek')}
      <div class="obrazok-popis">Histogram dĺžok za sebou idúcich poškodených vzoriek</div>
    </div>
    <h3 style="margin: 24px 0 12px; color:#b71c1c; font-size:1rem;">
      Zoznam poškodených vzoriek
    </h3>
    {tabulka_poskodennych(df)}
  </div>
</div>

<!-- ============================================================ -->
<!-- 5. ROČNÁ ANALÝZA ŽIARENIA -->
<!-- ============================================================ -->
<div class="sekcia" id="rocna">
  <div class="sekcia-hlavicka"><h2>5. Ročná analýza slnečného žiarenia</h2></div>
  <div class="sekcia-telo">
    <div class="vystup-text">{text_na_html(txt_rocna)}</div>
    <div class="obrazok-kontainer">
      {img_tag('rocna_analyza', 'Ročná analýza žiarenia')}
      <div class="obrazok-popis">Priemerné žiarenie počas roka s disperziou ±1σ</div>
    </div>
  </div>
</div>

<!-- ============================================================ -->
<!-- 6. MESAČNÁ DENNÁ ANALÝZA -->
<!-- ============================================================ -->
<div class="sekcia" id="mesiacna">
  <div class="sekcia-hlavicka"><h2>6. Denné profily žiarenia pre každý mesiac</h2></div>
  <div class="sekcia-telo">
    <div class="vystup-text">{text_na_html(txt_mesiacna)}</div>
    <div class="obrazok-kontainer">
      {img_tag('mesiacna_analyza', 'Mesačná denná analýza')}
      <div class="obrazok-popis">Priemerné žiarenie počas dňa pre každý mesiac s disperziou ±1σ</div>
    </div>
  </div>
</div>

<!-- ============================================================ -->
<!-- 7. KORELAČNÁ ANALÝZA -->
<!-- ============================================================ -->
<div class="sekcia" id="korelacie">
  <div class="sekcia-hlavicka"><h2>7. Korelačná analýza parametrov</h2></div>
  <div class="sekcia-telo">
    <div class="vystup-text">{text_na_html(txt_korelacie)}</div>
    <div class="dvojica">
      <div class="obrazok-kontainer">
        {img_tag('heatmapa', 'Korelačná matica')}
        <div class="obrazok-popis">Pearsonova korelačná matica</div>
      </div>
      <div class="obrazok-kontainer">
        {img_tag('scatter', 'Scatter plot korelácie')}
        <div class="obrazok-popis">Scatter ploty vybraných dvojíc parametrov</div>
      </div>
    </div>
  </div>
</div>


</main>

<!-- PÄTA -->
<footer>
  Vygenerované automaticky &nbsp;|&nbsp; Vstupný súbor: {csv_subor}
</footer>

</body>
</html>"""

    # ----------------------------------------------------------
    # Uloženie HTML súboru
    # ----------------------------------------------------------
    cesta_html = os.path.join(output_dir, 'report.html')   # Cesta k výstupu
    with open(cesta_html, 'w', encoding='utf-8') as f:     # Otvoríme súbor pre zápis (UTF-8 kvôli slovenčine)
        f.write(html)   # Zapíšeme celý HTML dokument

    velkost_kb = os.path.getsize(cesta_html) / 1024   # Vypočítame veľkosť súboru v kB
    print(f"  HTML report uložený: {cesta_html}  ({velkost_kb:.0f} kB)")   # Vypíšeme cestu a veľkosť
    return cesta_html   # Vrátime cestu k vygenerovanému HTML súboru
