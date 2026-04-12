# ==============================================================
# MODUL: Analýza slnečného žiarenia – ročné a mesačné denné profily
# ==============================================================

import os                       # Práca so súborovým systémom
import torch                    # PyTorch pre tenzorové štatistiky
import matplotlib.pyplot as plt  # Kreslenie grafov


# --------------------------------------------------------------
# 1. Ročná analýza – priemer a disperzia žiarenia počas roka
# --------------------------------------------------------------

def rocna_analyza_ziarenia(df, output_dir):
    """
    Pre každý deň v roku vypočíta priemerné žiarenie a jeho disperziu (std).
    Zobrazí ročný priebeh s pásom ±1σ.
    """
    print("\n=== ROČNÁ ANALÝZA ŽIARENIA ===")

    # Konverzia hodnôt žiarenia na PyTorch tenzor pre výpočet štatistík
    irr_tensor = torch.tensor(df['Irradiance'].dropna().values, dtype=torch.float32)

    # Výpis celkových štatistík pomocou PyTorch
    print(f"  PyTorch – celkový priemer:  {irr_tensor.mean().item():.2f} W/m²")
    print(f"  PyTorch – std odchýlka:     {irr_tensor.std().item():.2f} W/m²")
    print(f"  PyTorch – maximum:          {irr_tensor.max().item():.2f} W/m²")
    print(f"  PyTorch – minimum:          {irr_tensor.min().item():.2f} W/m²")

    # Skupinové štatistiky podľa dňa v roku (1–366)
    stats = df.groupby('DayOfYear')['Irradiance'].agg(['mean', 'std']).reset_index()
    stats.columns = ['DayOfYear', 'Priemer', 'Std']
    stats['Std'] = stats['Std'].fillna(0)   # Deň s jedinou vzorkou má std = NaN → nastavíme 0

    # --- Graf: Priemerné žiarenie počas roka s pásom disperzie ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Horný graf – priemer s pásom ±1σ
    ax1.plot(stats['DayOfYear'], stats['Priemer'], color='darkorange', linewidth=1.5,
             label='Priemerné žiarenie')
    ax1.fill_between(stats['DayOfYear'],
                     (stats['Priemer'] - stats['Std']).clip(lower=0),
                     stats['Priemer'] + stats['Std'],
                     alpha=0.3, color='orange', label='±1σ disperzia')
    ax1.set_xlabel('Deň v roku')
    ax1.set_ylabel('Žiarenie [W/m²]')
    ax1.set_title('Priemerné žiarenie počas roka s disperziou')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Dolný graf – štandardná odchýlka (disperzia) po dňoch
    ax2.plot(stats['DayOfYear'], stats['Std'], color='crimson', linewidth=1.5)
    ax2.set_xlabel('Deň v roku')
    ax2.set_ylabel('Štandardná odchýlka [W/m²]')
    ax2.set_title('Disperzia žiarenia počas roka')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rocna_analyza_ziarenia.png'), dpi=150)
    plt.close()
    print(f"  Graf uložený: rocna_analyza_ziarenia.png")


# --------------------------------------------------------------
# 2. Mesačná denná analýza – hodinové profily pre každý mesiac
# --------------------------------------------------------------

def denna_analyza_pre_mesiace(df, output_dir):
    """
    Pre každý mesiac zobrazí priemerný denný profil žiarenia (po hodinách)
    s pásom disperzie ±1σ.
    """
    print("\n=== DENNÁ ANALÝZA ŽIARENIA PRE KAŽDÝ MESIAC ===")

    # Slovenské názvy mesiacov pre grafy
    nazvy = ['Január', 'Február', 'Marec', 'Apríl', 'Máj', 'Jún',
             'Júl', 'August', 'September', 'Október', 'November', 'December']

    # Mriežka 3 × 4 grafov (jeden pre každý mesiac)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()   # Sploštenie 2D pola pre jednoduchú iteráciu

    for mesiac in range(1, 13):
        data_m = df[df['Month'] == mesiac]   # Dáta pre aktuálny mesiac

        if len(data_m) == 0:
            print(f"  {nazvy[mesiac - 1]}: žiadne dáta")
            continue

        # Výpis mesačných štatistík pomocou PyTorch tenzora
        irr = torch.tensor(data_m['Irradiance'].dropna().values, dtype=torch.float32)
        print(f"  {nazvy[mesiac - 1]:12s}: "
              f"priemer={irr.mean().item():7.1f} W/m², "
              f"std={irr.std().item():7.1f} W/m², "
              f"max={irr.max().item():7.1f} W/m², "
              f"vzorky={len(data_m):5d}")

        # Skupinové štatistiky podľa hodiny dňa
        hodinove = data_m.groupby('Hour')['Irradiance'].agg(['mean', 'std']).reset_index()
        hodinove.columns = ['Hodina', 'Priemer', 'Std']
        hodinove['Std'] = hodinove['Std'].fillna(0)

        # Kreslenie denného profilu pre tento mesiac
        ax = axes[mesiac - 1]
        ax.plot(hodinove['Hodina'], hodinove['Priemer'],
                color='darkorange', linewidth=2, marker='o', markersize=3)
        ax.fill_between(hodinove['Hodina'],
                        (hodinove['Priemer'] - hodinove['Std']).clip(lower=0),
                        hodinove['Priemer'] + hodinove['Std'],
                        alpha=0.3, color='orange')
        ax.set_title(nazvy[mesiac - 1])
        ax.set_xlabel('Hodina dňa')
        ax.set_ylabel('Žiarenie [W/m²]')
        ax.set_xlim(0, 23)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Priemerné žiarenie počas dňa pre každý mesiac (±1σ disperzia)',
                 fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'denna_analyza_mesiacov.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Graf uložený: denna_analyza_mesiacov.png")
