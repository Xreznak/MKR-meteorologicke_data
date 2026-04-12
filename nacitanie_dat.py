# ==============================================================
# MODUL: Načítanie a príprava meteorologických dát
# ==============================================================

import pandas as pd   # Práca s tabuľkovými dátami


def nacitaj_data(cesta_k_suboru):
    """
    Načíta CSV súbor a pripraví dáta na analýzu.
    Vracia: pandas DataFrame s parsovanými časovými značkami a pomocnými stĺpcami.
    """
    # Načítanie CSV súboru do tabuľky DataFrame
    df = pd.read_csv(cesta_k_suboru)

    # Stĺpec DateTime má formát "MM/DD/YYYY HH:MM:SS.mmm#číslo_snímky"
    # Odstraňujeme časť za '#', ktorá označuje číslo snímky (nie je súčasťou času)
    df['DateTime'] = df['DateTime'].str.split('#').str[0]

    # Konverzia reťazca na datetime objekt – format='mixed' zvládne záznamy
    # aj bez milisekúnd, dayfirst=False = americký formát MM/DD/YYYY
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed', dayfirst=False)

    # Vytvorenie pomocného stĺpca s len dátumom (bez hodiny a minút)
    df['Date'] = df['DateTime'].dt.date

    # Vytvorenie stĺpca pre hodinu dňa (0–23) – potrebné pre denné profily
    df['Hour'] = df['DateTime'].dt.hour

    # Vytvorenie stĺpca pre mesiac (1 = január, 12 = december)
    df['Month'] = df['DateTime'].dt.month

    # Vytvorenie stĺpca pre deň v roku (1–366) – potrebné pre ročné grafy
    df['DayOfYear'] = df['DateTime'].dt.dayofyear

    # Zoradenie všetkých záznamov chronologicky (podľa času merania)
    df = df.sort_values('DateTime').reset_index(drop=True)

    return df
