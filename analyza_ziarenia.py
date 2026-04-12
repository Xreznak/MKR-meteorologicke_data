# ==============================================================
# MODUL: Analýza slnečného žiarenia – ročné a mesačné denné profily
# ==============================================================

import os                        # Práca so súborovým systémom
import torch                     # PyTorch pre tenzorové štatistiky
import matplotlib.pyplot as plt  # Kreslenie grafov


# --------------------------------------------------------------
# 1. Ročná analýza – priemer a disperzia žiarenia počas roka
# --------------------------------------------------------------

def rocna_analyza_ziarenia(df, output_dir):
    """
    Pre každý deň v roku vypočíta priemerné žiarenie a jeho disperziu (std).
    Zobrazí ročný priebeh s pásom ±1σ.
    """
    print("\n=== ROČNÁ ANALÝZA ŽIARENIA ===")   # Nadpis sekcie

    # Konverzia hodnôt žiarenia na PyTorch tenzor pre výpočet štatistík
    irr_tensor = torch.tensor(df['Irradiance'].dropna().values, dtype=torch.float32)   # Tenzor žiarenia

    # Výpis celkových štatistík pomocou PyTorch
    print(f"  PyTorch – celkový priemer:  {irr_tensor.mean().item():.2f} W/m²")   # Priemer
    print(f"  PyTorch – std odchýlka:     {irr_tensor.std().item():.2f} W/m²")    # Štandardná odchýlka
    print(f"  PyTorch – maximum:          {irr_tensor.max().item():.2f} W/m²")    # Maximum
    print(f"  PyTorch – minimum:          {irr_tensor.min().item():.2f} W/m²")    # Minimum

    # Skupinové štatistiky podľa dňa v roku (1–366)
    stats = df.groupby('DayOfYear')['Irradiance'].agg(['mean', 'std']).reset_index()   # Priemer a std pre každý deň
    stats.columns = ['DayOfYear', 'Priemer', 'Std']                   # Premenujeme stĺpce
    stats['Std'] = stats['Std'].fillna(0)   # Deň s jedinou vzorkou má std = NaN → nastavíme 0

    # --- Graf: Priemerné žiarenie počas roka s pásom disperzie ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))   # Dva grafy pod sebou

    # Horný graf – priemer s pásom ±1σ
    ax1.plot(stats['DayOfYear'], stats['Priemer'], color='darkorange', linewidth=1.5,
             label='Priemerné žiarenie')                               # Čiara priemeru
    ax1.fill_between(stats['DayOfYear'],
                     (stats['Priemer'] - stats['Std']).clip(lower=0), # Spodná hranica pásu (min 0)
                     stats['Priemer'] + stats['Std'],                  # Horná hranica pásu
                     alpha=0.3, color='orange', label='±1σ disperzia') # Vyplnený pás disperzie
    ax1.set_xlabel('Deň v roku')                        # Popis osi x
    ax1.set_ylabel('Žiarenie [W/m²]')                   # Popis osi y
    ax1.set_title('Priemerné žiarenie počas roka s disperziou')   # Nadpis
    ax1.legend()                                        # Legenda
    ax1.grid(True, alpha=0.3)                           # Mriežka

    # Dolný graf – štandardná odchýlka (disperzia) po dňoch
    ax2.plot(stats['DayOfYear'], stats['Std'], color='crimson', linewidth=1.5)   # Čiara std
    ax2.set_xlabel('Deň v roku')                        # Popis osi x
    ax2.set_ylabel('Štandardná odchýlka [W/m²]')        # Popis osi y
    ax2.set_title('Disperzia žiarenia počas roka')      # Nadpis
    ax2.grid(True, alpha=0.3)                           # Mriežka

    plt.tight_layout()                                  # Prispôsobenie okrajov
    plt.savefig(os.path.join(output_dir, 'rocna_analyza_ziarenia.png'), dpi=150)   # Uloženie
    plt.close()                                         # Zatvoríme graf
    print(f"  Graf uložený: rocna_analyza_ziarenia.png")   # Potvrdenie


# --------------------------------------------------------------
# 2. Mesačná denná analýza – hodinové profily pre každý mesiac
# --------------------------------------------------------------

def denna_analyza_pre_mesiace(df, output_dir):
    """
    Pre každý mesiac zobrazí priemerný denný profil žiarenia (po hodinách)
    s pásom disperzie ±1σ.
    """
    print("\n=== DENNÁ ANALÝZA ŽIARENIA PRE KAŽDÝ MESIAC ===")   # Nadpis sekcie

    # Slovenské názvy mesiacov pre grafy
    nazvy = ['Január', 'Február', 'Marec', 'Apríl', 'Máj', 'Jún',
             'Júl', 'August', 'September', 'Október', 'November', 'December']

    # Mriežka 3 × 4 grafov (jeden pre každý mesiac)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))   # 12 grafov v mriežke 3 riadky × 4 stĺpce
    axes = axes.flatten()   # Sploštenie 2D poľa na 1D pre jednoduchú iteráciu

    for mesiac in range(1, 13):   # Iterujeme cez všetky mesiace (1 = január, 12 = december)
        data_m = df[df['Month'] == mesiac]   # Vyfiltrujeme dáta pre aktuálny mesiac

        if len(data_m) == 0:                 # Ak pre daný mesiac nie sú žiadne dáta
            print(f"  {nazvy[mesiac - 1]}: žiadne dáta")   # Informácia o prázdnom mesiaci
            continue                          # Preskočíme na ďalší mesiac

        # Výpis mesačných štatistík pomocou PyTorch tenzora
        irr = torch.tensor(data_m['Irradiance'].dropna().values, dtype=torch.float32)   # Tenzor žiarenia
        print(f"  {nazvy[mesiac - 1]:12s}: "
              f"priemer={irr.mean().item():7.1f} W/m², "
              f"std={irr.std().item():7.1f} W/m², "
              f"max={irr.max().item():7.1f} W/m², "
              f"vzorky={len(data_m):5d}")      # Výpis štatistík pre daný mesiac

        # Skupinové štatistiky podľa hodiny dňa
        hodinove = data_m.groupby('Hour')['Irradiance'].agg(['mean', 'std']).reset_index()   # Priemer a std po hodinách
        hodinove.columns = ['Hodina', 'Priemer', 'Std']   # Premenujeme stĺpce
        hodinove['Std'] = hodinove['Std'].fillna(0)        # Hodina s jedinou vzorkou → std = 0

        # Kreslenie denného profilu pre tento mesiac
        ax = axes[mesiac - 1]                              # Vyberieme správny subplot
        ax.plot(hodinove['Hodina'], hodinove['Priemer'],
                color='darkorange', linewidth=2, marker='o', markersize=3,
                label='Priemer')                           # Čiara priemeru s bodmi
        ax.fill_between(hodinove['Hodina'],
                        (hodinove['Priemer'] - hodinove['Std']).clip(lower=0),   # Spodná hranica (min 0)
                        hodinove['Priemer'] + hodinove['Std'],                   # Horná hranica
                        alpha=0.3, color='orange', label='±1σ disperzia')        # Vyplnený pás disperzie
        ax.set_title(nazvy[mesiac - 1])                    # Nadpis grafu = názov mesiaca
        ax.set_xlabel('Hodina dňa')                        # Popis osi x
        ax.set_ylabel('Žiarenie [W/m²]')                   # Popis osi y
        ax.set_xlim(0, 23)                                 # Rozsah osi x: 0 až 23 hodín
        ax.legend(fontsize=7)                              # Legenda s menším písmom (zmestí sa)
        ax.grid(True, alpha=0.3)                           # Mriežka

    plt.suptitle('Priemerné žiarenie počas dňa pre každý mesiac (±1σ disperzia)',
                 fontsize=16, y=1.01)   # Spoločný nadpis pre všetky grafy
    plt.tight_layout()                  # Prispôsobenie okrajov
    plt.savefig(os.path.join(output_dir, 'denna_analyza_mesiacov.png'), dpi=150,
                bbox_inches='tight')    # Uloženie (bbox_inches='tight' zahrnie aj suptitle)
    plt.close()                         # Zatvoríme graf
    print(f"  Graf uložený: denna_analyza_mesiacov.png")   # Potvrdenie
