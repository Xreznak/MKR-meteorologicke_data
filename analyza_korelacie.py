# ==============================================================
# MODUL: Korelačná analýza a súhrnné štatistiky parametrov
# ==============================================================

import os                        # Práca so súborovým systémom
import torch                     # PyTorch pre tenzorové výpočty
import matplotlib.pyplot as plt  # Kreslenie grafov
import seaborn as sns            # Heatmapa korelačnej matice


# Zoznam parametrov, ktoré chceme analyzovať
PARAMETRE = [
    'Irradiance', 'BodyTemperature', 'RelativeHumidity',
    'HumidityTemp', 'Pressure', 'TiltAngle', 'FanSpeed',
    'SunAzimuth', 'SunZenith'
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
    fig, ax = plt.subplots(figsize=(11, 9))   # Nová figúra
    sns.heatmap(kor_matica,
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

    # --- Graf: Scatter ploty pre vybrané dvojice parametrov ---
    df_sc = df[['Irradiance', 'SunZenith', 'BodyTemperature', 'Pressure']].dropna()   # Vyberieme stĺpce
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))   # Tri grafy vedľa seba

    # Žiarenie vs Zenitový uhol slnka (fyzikálne priama závislosť)
    axes[0].scatter(df_sc['SunZenith'], df_sc['Irradiance'], alpha=0.1, s=2, color='darkorange')
    axes[0].set_xlabel('Zenitový uhol slnka [°]')   # Popis osi x
    axes[0].set_ylabel('Žiarenie [W/m²]')            # Popis osi y
    axes[0].set_title('Žiarenie vs Zenitový uhol')   # Nadpis
    axes[0].grid(True, alpha=0.3)                    # Mriežka

    # Žiarenie vs Teplota senzora
    axes[1].scatter(df_sc['BodyTemperature'], df_sc['Irradiance'], alpha=0.1, s=2, color='crimson')
    axes[1].set_xlabel('Teplota senzora [°C]')       # Popis osi x
    axes[1].set_ylabel('Žiarenie [W/m²]')            # Popis osi y
    axes[1].set_title('Žiarenie vs Teplota senzora') # Nadpis
    axes[1].grid(True, alpha=0.3)                    # Mriežka

    # Teplota senzora vs Atmosferický tlak
    axes[2].scatter(df_sc['Pressure'], df_sc['BodyTemperature'], alpha=0.1, s=2, color='seagreen')
    axes[2].set_xlabel('Atmosferický tlak [hPa]')    # Popis osi x
    axes[2].set_ylabel('Teplota senzora [°C]')       # Popis osi y
    axes[2].set_title('Teplota senzora vs Tlak')     # Nadpis
    axes[2].grid(True, alpha=0.3)                    # Mriežka

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
