# ==============================================================
# MODUL: Analýza chýb – chýbajúce dni, chýbajúce a poškodené vzorky
# ==============================================================

import os               # Práca so súborovým systémom (cesty, adresáre)
import numpy as np      # Numerické výpočty
import pandas as pd     # Práca s tabuľkovými dátami
import matplotlib.pyplot as plt   # Kreslenie grafov
import matplotlib.dates as mdates  # Formátovanie dátumov na osiach


# --------------------------------------------------------------
# 1. Chýbajúce dni
# --------------------------------------------------------------

def analyza_chybajucich_dni(df, output_dir):
    """
    Nájde dni, v ktorých chýbajú VŠETKY merania (celé vynechané dni).
    Uloží histogramy.
    """
    print("\n=== ANALÝZA CHÝBAJÚCICH DNÍ ===")

    # Nájdenie najstaršieho a najnovšieho dátumu v dátach
    min_datum = df['DateTime'].min().date()
    max_datum = df['DateTime'].max().date()

    # Generovanie zoznamu každého dňa od začiatku do konca merania
    vsetky_dni = pd.date_range(start=min_datum, end=max_datum, freq='D').date

    # Zistenie, ktoré dni sa skutočne nachádzajú v dátach
    pritomne_dni = set(df['Date'].unique())

    # Odčítaním množín nájdeme chýbajúce dni
    chybajuce_dni = sorted(set(vsetky_dni) - pritomne_dni)

    # Výpis štatistík
    print(f"  Rozsah merania:              {min_datum} až {max_datum}")
    print(f"  Celkový počet dní v rozsahu: {len(vsetky_dni)}")
    print(f"  Počet dní s dátami:          {len(pritomne_dni)}")
    print(f"  Počet chýbajúcich dní:       {len(chybajuce_dni)}")
    if chybajuce_dni:
        print(f"  Chýbajúce dni:               {chybajuce_dni}")

    # --- Graf: Histogram chýbajúcich dní podľa mesiaca ---
    if chybajuce_dni:
        # Extrakcia mesiaca z každého chýbajúceho dňa
        mesiace_chyb = [d.month for d in chybajuce_dni]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(mesiace_chyb, bins=range(1, 14), align='left', rwidth=0.8,
                color='tomato', edgecolor='black')
        ax.set_xlabel('Mesiac')
        ax.set_ylabel('Počet chýbajúcich dní')
        ax.set_title('Histogram chýbajúcich dní podľa mesiaca')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'Máj', 'Jún',
                            'Júl', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'histogram_chybajucich_dni.png'), dpi=150)
        plt.close()
        print(f"  Graf uložený: histogram_chybajucich_dni.png")

    # --- Graf: Počet vzoriek za každý deň (histogram distribúcie) ---
    vzorky_za_den = df.groupby('Date').size()   # Spočítanie vzoriek pre každý deň

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(vzorky_za_den.values, bins=30, color='steelblue', edgecolor='black')
    ax.axvline(vzorky_za_den.mean(), color='red', linestyle='--',
               label=f'Priemer: {vzorky_za_den.mean():.1f}')
    ax.set_xlabel('Počet vzoriek za deň')
    ax.set_ylabel('Počet dní')
    ax.set_title('Distribúcia počtu vzoriek na deň')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histogram_vzoriek_za_den.png'), dpi=150)
    plt.close()
    print(f"  Graf uložený: histogram_vzoriek_za_den.png")
    print(f"  Priemerný počet vzoriek za deň: {vzorky_za_den.mean():.1f}")
    print(f"  Min / Max vzoriek za deň:       {vzorky_za_den.min()} / {vzorky_za_den.max()}")

    # --- Graf: Časový priebeh počtu vzoriek za deň ---
    vzorky_ts = vzorky_za_den.copy()
    vzorky_ts.index = pd.to_datetime(vzorky_ts.index)   # Konverzia indexu na datetime

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(vzorky_ts.index, vzorky_ts.values, color='steelblue', linewidth=0.8)
    ax.set_xlabel('Dátum')
    ax.set_ylabel('Počet vzoriek za deň')
    ax.set_title('Počet vzoriek za každý deň v čase')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vzorky_za_den_casovy_priebeh.png'), dpi=150)
    plt.close()
    print(f"  Graf uložený: vzorky_za_den_casovy_priebeh.png")

    return chybajuce_dni


# --------------------------------------------------------------
# 2. Chýbajúce vzorky (NaN hodnoty)
# --------------------------------------------------------------

def analyza_chybajucich_vzoriek(df):
    """
    Spočíta chýbajúce (NaN) hodnoty pre každý numerický stĺpec.
    """
    print("\n=== ANALÝZA CHÝBAJÚCICH VZORIEK (NaN HODNOTY) ===")

    # Zoznam stĺpcov, ktoré chceme skontrolovať
    stlpce = ['Irradiance', 'IrradianceNotCompensated', 'BodyTemperature',
              'RelativeHumidity', 'HumidityTemp', 'Pressure', 'PressureAvg',
              'TiltAngle', 'FanSpeed', 'SunAzimuth', 'SunZenith']

    # Výpočet počtu NaN hodnôt pre každý stĺpec
    chybajuce = df[stlpce].isnull().sum()
    print(f"  Celkový počet vzoriek: {len(df)}")
    for stlpec, pocet in chybajuce.items():
        # Výpis aj stĺpcov bez chýb, aby bolo jasné, že boli skontrolované
        stav = f"{pocet} ({100 * pocet / len(df):.2f}%)" if pocet > 0 else "0 chýbajúcich"
        print(f"    {stlpec:30s}: {stav}")

    # Celkový počet riadkov s aspoň jednou NaN hodnotou
    riadky_s_nan = df[stlpce].isnull().any(axis=1).sum()
    print(f"  Riadkov s aspoň jednou chýbajúcou hodnotou: {riadky_s_nan}")


# --------------------------------------------------------------
# 3. Poškodené vzorky (fyzikálne medze)
# --------------------------------------------------------------

def detekcia_poskodennych_vzoriek(df):
    """
    Detekuje vzorky, kde hodnota je mimo fyzikálne možného rozsahu.
    Vracia: boolovskú masku poškodených riadkov a pravdepodobnosť chybnej vzorky.
    """
    print("\n=== DETEKCIA POŠKODENÝCH VZORIEK (FYZIKÁLNE MEDZE) ===")

    # Fyzikálne medze (min, max) pre každý parameter
    fyzikalne_medze = {
        'Irradiance':               (-5,   1500),   # W/m²  – slnečné žiarenie
        'IrradianceNotCompensated': (-5,   1500),   # W/m²
        'BodyTemperature':          (-40,  80),     # °C    – teplota senzora
        'RelativeHumidity':         (0,    100),    # %     – relatívna vlhkosť
        'HumidityTemp':             (-40,  60),     # °C    – teplota vlhkomeru
        'Pressure':                 (800,  1100),   # hPa   – atmosferický tlak
        'PressureAvg':              (800,  1100),   # hPa
        'TiltAngle':                (-90,  90),     # °     – uhol naklonenia
        'FanSpeed':                 (0,    15000),  # RPM   – rýchlosť ventilátora
        'SunZenith':                (0,    180),    # °     – zenitový uhol slnka
        'SunAzimuth':               (0,    360),    # °     – azimut slnka
    }

    # Začíname s maskou, kde všetko je False (žiadna vzorka nie je poškodená)
    poskodene_maska = pd.Series(False, index=df.index)

    for stlpec, (min_val, max_val) in fyzikalne_medze.items():
        if stlpec not in df.columns:
            continue
        # Vzorky mimo medzi, ktoré zároveň nie sú NaN
        mimo_medze = ((df[stlpec] < min_val) | (df[stlpec] > max_val)) & df[stlpec].notna()
        pocet = mimo_medze.sum()
        stav = f"{pocet} ({100 * pocet / len(df):.3f}%)" if pocet > 0 else "0 poškodených"
        print(f"    {stlpec:30s}: {stav}")
        # Pridáme do celkovej masky (OR – stačí jedna poškodená hodnota v riadku)
        poskodene_maska = poskodene_maska | mimo_medze

    celkove = poskodene_maska.sum()               # Celkový počet poškodených riadkov
    pravdepodobnost = celkove / len(df)           # Pravdepodobnosť chybnej vzorky

    print(f"\n  Celkový počet poškodených vzoriek: {celkove}")
    print(f"  Pravdepodobnosť chybnej vzorky:    {pravdepodobnost:.6f} ({100 * pravdepodobnost:.4f}%)")

    return poskodene_maska, pravdepodobnost


# --------------------------------------------------------------
# 4. Histogramy dĺžok za sebou idúcich chýb
# --------------------------------------------------------------

def histogramy_dlzok_chyb(df, poskodene_maska, output_dir):
    """
    Pre chýbajúce (NaN) aj poškodené vzorky vypočíta dĺžky súvislých skupín
    a zobrazí ich ako histogram.
    """
    print("\n=== HISTOGRAMY DĹŽOK ZA SEBOU IDÚCICH CHÝB ===")

    # Spustíme analýzu pre oba typy chýb
    chyby = {
        'chybajucich_nan':  (df['Irradiance'].isnull(), 'chýbajúcich (NaN) vzoriek'),
        'poskodennych':     (poskodene_maska,             'poškodených vzoriek'),
    }

    for nazov_suboru, (maska, popis) in chyby.items():
        # Výpočet dĺžok súvislých skupín True hodnôt
        skupiny = []
        dlzka = 0
        for hodnota in maska:
            if hodnota:
                dlzka += 1       # Rozširujeme aktuálnu skupinu
            else:
                if dlzka > 0:
                    skupiny.append(dlzka)   # Skupina skončila, uložíme jej dĺžku
                dlzka = 0
        if dlzka > 0:
            skupiny.append(dlzka)   # Posledná skupina (ak séria končí chybou)

        print(f"  --- {popis} ---")
        if skupiny:
            print(f"    Počet skupín:         {len(skupiny)}")
            print(f"    Max dĺžka skupiny:    {max(skupiny)}")
            print(f"    Priemerná dĺžka:      {np.mean(skupiny):.2f}")
            # Kreslenie histogramu dĺžok skupín
            fig, ax = plt.subplots(figsize=(10, 5))
            # Half-integer hrany: hodnota 1 → bin [0.5, 1.5], hodnota 2 → [1.5, 2.5], ...
            # Tak je každé celé číslo vycentrované vo vlastnom stĺpci
            max_dlzka = min(max(skupiny), 30)
            bin_edges = [i + 0.5 for i in range(0, max_dlzka + 1)]   # [0.5, 1.5, 2.5, ...]
            ax.hist(skupiny, bins=bin_edges, color='coral', edgecolor='black')
            ax.set_xticks(range(1, max_dlzka + 1))   # Popisky osi x len na celých číslach
            ax.set_xlabel('Dĺžka súvislej skupiny (počet vzoriek)')
            ax.set_ylabel('Počet skupín')
            ax.set_title(f'Histogram dĺžok za sebou idúcich {popis}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'histogram_dlzok_{nazov_suboru}.png'), dpi=150)
            plt.close()
            print(f"    Graf uložený: histogram_dlzok_{nazov_suboru}.png")
        else:
            print(f"    Žiadne za sebou idúce {popis} neboli nájdené.")
