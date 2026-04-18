# ==============================================================
# MODUL: Korelačná analýza a súhrnné štatistiky parametrov
# ==============================================================

import os                        # Práca so súborovým systémom
import torch                     # PyTorch pre tenzorové výpočty
import matplotlib.pyplot as plt  # Kreslenie grafov
import seaborn as sns            # Heatmapa korelačnej matice


# Slovenské názvy stĺpcov pre popisky grafov
NAZVY_SK = {
    'Irradiance':               'Žiarenie [W/m²]',
    'IrradianceNotCompensated': 'Žiarenie nekompenz. [W/m²]',
    'BodyTemperature':          'Teplota senzora [°C]',
    'RelativeHumidity':         'Relatívna vlhkosť [%]',
    'HumidityTemp':             'Teplota vlhkomera [°C]',
    'Pressure':                 'Atmosf. tlak [hPa]',
    'PressureAvg':              'Atmosf. tlak – priemer [hPa]',
    'PressureTemp':             'Teplota tlakomera [°C]',
    'PressureTempAvg':          'Teplota tlakomera – priemer [°C]',
    'TiltAngle':                'Uhol naklonenia [°]',
    'TiltAngleAvg':             'Uhol naklonenia – priemer [°]',
    'FanSpeed':                 'Rýchlosť ventilátora [RPM]',
    'HeaterCurrent':            'Prúd ohrievača [A]',
    'FanCurrent':               'Prúd ventilátora [A]',
    'SunLatitude':              'Zem. šírka stanice [°]',
    'SunLongitude':             'Zem. dĺžka stanice [°]',
    'SunAzimuth':               'Azimut slnka [°]',
    'SunZenith':                'Zenitový uhol slnka [°]',
}


# Zoznam parametrov pre korelačnú analýzu.
# SunLatitude a SunLongitude sú vynechané – sú konštantné (jedna hodnota celý dataset),
# korelácia konštanty s čímkoľvek je matematicky nedefinovaná (std = 0).
PARAMETRE = [
    'Irradiance', 'IrradianceNotCompensated',
    'BodyTemperature', 'RelativeHumidity', 'HumidityTemp',
    'Pressure', 'PressureAvg', 'PressureTemp', 'PressureTempAvg',
    'TiltAngle', 'TiltAngleAvg', 'FanSpeed',
    'HeaterCurrent', 'FanCurrent',
    'SunAzimuth', 'SunZenith',
]


# --------------------------------------------------------------
# 1. Korelačná analýza
# --------------------------------------------------------------

def korelacie_parametrov(df, output_dir):
    """
    Vypočíta Pearsonove korelačné koeficienty medzi všetkými parametrami.
    Overí výsledky pomocou PyTorch. Uloží heatmapu a scatter ploty.
    """
    print("\n=== KORELAČNÁ ANALÝZA PARAMETROV ===")   # Nadpis sekcie

    # Výpočet korelačnej matice pomocou pandas (Pearsonova metóda)
    kor_matica = df[PARAMETRE].corr(method='pearson')   # Matica korelácie všetkých párov

    # Výpis korelácie každého parametra s Irradiance – zoradené podľa absolútnej hodnoty
    print("  Korelácie ostatných parametrov s Irradiance (žiarením):")
    kor_s_irr = kor_matica['Irradiance'].drop('Irradiance').sort_values(key=abs, ascending=False)   # Zoradenie
    for param, r in kor_s_irr.items():   # Iterujeme cez každý parameter
        sila = "silná" if abs(r) > 0.5 else ("stredná" if abs(r) > 0.3 else "slabá")   # Sila korelácie
        smer = "pozitívna" if r > 0 else "negatívna"                                    # Smer korelácie
        print(f"    {param:30s}: r = {r:+.4f}  ({sila} {smer})")                        # Výpis riadku

    # --- Overenie korelačného výpočtu pomocou PyTorch tenzorov ---
    df_c = df[PARAMETRE].dropna()   # Odstraňujeme riadky s NaN
    if len(df_c) > 1:               # Pokračujeme len ak máme aspoň 2 riadky
        T = torch.tensor(df_c.values, dtype=torch.float32)   # Konverzia celej tabuľky na tenzor

        # Vyberieme stĺpce Irradiance a SunZenith
        irr = T[:, 0]                                   # Stĺpec Irradiance (index 0)
        zen = T[:, PARAMETRE.index('SunZenith')]         # Stĺpec SunZenith

        # Pearsonova korelácia: odchýlky od priemeru → ich skalárny súčin / súčin veľkostí
        irr_c = irr - irr.mean()                         # Odchýlky Irradiance od priemeru
        zen_c = zen - zen.mean()                         # Odchýlky SunZenith od priemeru
        r = (irr_c * zen_c).sum() / (irr_c.norm() * zen_c.norm())   # Výsledná korelácia

        print(f"\n  PyTorch – Irradiance vs SunZenith: {r.item():.4f}")   # Výpis výsledku

    # --- Graf: Heatmapa korelačnej matice ---
    # Premenujeme indexy a stĺpce matice na slovenské skrátené názvy
    nazvy_kratke = {k: v.split(' [')[0] for k, v in NAZVY_SK.items()}   # Bez jednotky (kratšie)
    kor_matica_sk = kor_matica.rename(index=nazvy_kratke, columns=nazvy_kratke)

    fig, ax = plt.subplots(figsize=(14, 11))   # Nová figúra (väčšia kvôli dlhším názvom)
    sns.heatmap(kor_matica_sk,
                annot=True,        # Zobrazenie hodnôt v každej bunke
                fmt='.2f',         # Formát: 2 desatinné miesta
                cmap='coolwarm',   # Modrá = negatívna, červená = pozitívna korelácia
                center=0,          # Biela farba pri r = 0
                vmin=-1, vmax=1,   # Rozsah farebnej škály
                square=True,       # Štvorcové bunky
                linewidths=0.5,    # Tenká mriežka medzi bunkami
                ax=ax)             # Kreslíme do nášho subplotu
    ax.set_title('Korelačná matica meteorologických parametrov (Pearson)')   # Nadpis
    plt.tight_layout()             # Prispôsobenie okrajov
    plt.savefig(os.path.join(output_dir, 'heatmapa_korelacie.png'), dpi=150)   # Uloženie
    plt.close()                    # Zatvoríme graf
    print(f"  Graf uložený: heatmapa_korelacie.png")   # Potvrdenie

    # --- Graf: Scatter ploty ---
    # Riadok 1 (pevný): trojuholník Žiarenie / Atmosf. tlak / Teplota senzora
    #   – ukazuje že Pressure a BodyTemperature sú navzájom vysoko korelované
    #     (sezónny efekt) a oba súvisia s Irradiance
    # Riadok 2+: auto-vybrané najsilnejšie rôznorodé páry zo zvyšných parametrov

    # Pevných 9 párov – tri trojuholníky, každý vysvetľuje vzťahy v trojici parametrov:
    #   Trojuholník 1: Žiarenie / Atmosf. tlak / Teplota senzora  (sezónny efekt)
    #   Trojuholník 2: Atmosf. tlak / Rýchlosť ventilátora / Teplota tlakomera
    #   Trojuholník 3: Žiarenie / Zenitový uhol slnka / Azimut slnka
    PEVNE_PARY = [
        ('Irradiance', 'Pressure'),          # 1. Žiarenie vs Atmosf. tlak
        ('Irradiance', 'BodyTemperature'),   # 2. Žiarenie vs Teplota senzora
        ('Pressure',   'BodyTemperature'),   # 3. Atmosf. tlak vs Teplota senzora

        ('Irradiance', 'SunZenith'),         # 4. Žiarenie vs Zenitový uhol slnka
        ('Irradiance', 'SunAzimuth'),        # 5. Žiarenie vs Azimut slnka
        ('SunZenith',  'SunAzimuth'),        # 6. Zenitový uhol slnka vs Azimut slnka

        ('Pressure',        'FanSpeed'),        # 7. Atmosf. tlak vs Rýchlosť ventilátora
        ('Pressure',        'PressureTemp'),    # 8. Atmosf. tlak vs Teplota tlakomera
        ('FanSpeed',        'PressureTemp'),    # 9. Rýchlosť ventilátora vs Teplota tlakomera

        ('BodyTemperature', 'PressureTemp'),    # 10. Teplota senzora vs Teplota tlakomera
        ('BodyTemperature', 'HumidityTemp'),    # 11. Teplota senzora vs Teplota vlhkomera
        ('PressureTemp',    'HumidityTemp'),    # 12. Teplota tlakomera vs Teplota vlhkomera
    ]

    top_pary = [(abs(kor_matica.loc[p1, p2]), p1, p2) for p1, p2 in PEVNE_PARY]

    print(f"\n  Vybrané páry pre scatter ploty ({len(top_pary)} celkovo):")
    for r_val, p1, p2 in top_pary:
        smer = "pozitívna" if kor_matica.loc[p1, p2] > 0 else "negatívna"
        print(f"    {p1} vs {p2}: r = {kor_matica.loc[p1, p2]:+.4f}  ({smer})")

    # Kreslenie scatter plotov – mriežka 3×3 (alebo menej ak párov nie je dosť)
    n = len(top_pary)
    ncols = 3
    nrows = (n + ncols - 1) // ncols   # Zaokrúhlenie nahor
    paleta = ['darkorange', 'crimson', 'seagreen', 'steelblue',
              'mediumpurple', 'saddlebrown', 'teal', 'tomato', 'slategray',
              'goldenrod', 'indianred', 'cadetblue']
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten()   # Sploštíme na 1D zoznam pre jednoduchú iteráciu

    for ax, (r_val, p1, p2), farba in zip(axes, top_pary, paleta):
        df_sc = df[[p1, p2]].dropna()                          # Vyberieme len tieto dva stĺpce, bez NaN
        ax.scatter(df_sc[p2], df_sc[p1],
                   alpha=0.1, s=2, color=farba)                # Bodový graf (malé body, priesvitné)
        ax.set_xlabel(NAZVY_SK.get(p2, p2))                    # Slovenský popis osi x
        ax.set_ylabel(NAZVY_SK.get(p1, p1))                    # Slovenský popis osi y
        r_sign = kor_matica.loc[p1, p2]
        nazov1 = NAZVY_SK.get(p1, p1).split(' [')[0]          # Názov bez jednotky pre nadpis
        nazov2 = NAZVY_SK.get(p2, p2).split(' [')[0]
        ax.set_title(f'{nazov1}\nvs {nazov2}\nr = {r_sign:+.3f}')   # Nadpis s hodnotou r
        ax.grid(True, alpha=0.3)                               # Mriežka

    # Skryjeme prázdne subploty ak párov je menej ako 9
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()   # Prispôsobenie okrajov
    plt.savefig(os.path.join(output_dir, 'scatter_korelacie.png'), dpi=150)   # Uloženie
    plt.close()          # Zatvoríme graf
    print(f"  Graf uložený: scatter_korelacie.png")   # Potvrdenie


# --------------------------------------------------------------
# 2. Súhrnné štatistiky
# --------------------------------------------------------------

def suhrne_statistiky(df, output_dir):
    """
    Vypíše základné opisné štatistiky (min, max, priemer, std) pre všetky parametre.
    Uloží ich do CSV súboru. Overí pomocou PyTorch.
    """
    print("\n=== SÚHRNNÉ ŠTATISTIKY ===")   # Nadpis sekcie

    # Pandas describe() – count, mean, std, min, percentily, max
    stats = df[PARAMETRE].describe()    # Vypočítame štatistiky pre všetky parametre
    print(stats.to_string())            # Výpis tabuľky štatistík

    # Uloženie štatistík do CSV
    stats.to_csv(os.path.join(output_dir, 'suhrne_statistiky.csv'))   # Uložíme do CSV súboru
    print(f"\n  Uložené do: suhrne_statistiky.csv")   # Potvrdenie uloženia

    # --- PyTorch overenie štatistík ---
    print("\n  PyTorch – min | max | priemer | std:")
    df_c = df[PARAMETRE].dropna()                                       # Odstraňujeme riadky s NaN
    T = torch.tensor(df_c.values, dtype=torch.float32)                  # Celá tabuľka ako tenzor
    for i, param in enumerate(PARAMETRE):                               # Pre každý parameter
        col = T[:, i]                                                    # Vyberieme i-tý stĺpec
        print(f"    {param:30s}: "
              f"min={col.min().item():9.3f} | max={col.max().item():9.3f} | "
              f"mean={col.mean().item():9.3f} | std={col.std().item():9.3f}")   # Výpis štatistík
