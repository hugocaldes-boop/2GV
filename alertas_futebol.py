#!/usr/bin/env python3
"""
Alertas diários de futebol (BTTS + resultados raros) por Telegram.

Replica o "Modelo de Poisson simplificado corrigido" da folha Google Sheets:
  1. Golos esperados = (marcados x sofridos do adversário) / média da liga
  2. Ajuste de vantagem de casa (+5%), mantendo o total de golos
  3. Matriz de Poisson 0..10 x 0..10
  4. Células 0-0, 1-0, 0-1, 1-1 passam a ser a frequência histórica (= Poisson x alfa)
  5. As restantes células são multiplicadas por beta, calculado automaticamente
     para a matriz somar 100%

Uso:
  python alertas_futebol.py                          # corre os jogos de hoje e envia para o Telegram
  python alertas_futebol.py --sem-telegram           # só imprime
  python alertas_futebol.py --simular SP1 Barcelona Villarreal   # um jogo à escolha (para validar com a folha)

Variáveis de ambiente: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import argparse
import io
import os
from datetime import date
from math import exp, factorial

import numpy as np
import pandas as pd
import requests

# ============================ CONFIGURAÇÃO ============================
LIGAS = {
    "E0": "Premier League", "E1": "Championship",
    "SP1": "La Liga", "SP2": "La Liga 2",
    "D1": "Bundesliga", "D2": "2. Bundesliga",
    "I1": "Serie A", "I2": "Serie B",
    "F1": "Ligue 1", "F2": "Ligue 2",
    "N1": "Eredivisie", "P1": "Liga Portugal",
    "B1": "Liga Belga", "SC0": "Scottish Premiership",
    "T1": "Süper Lig", "G1": "Super League Grécia",
}

JOGOS_AMOSTRA = 10      # últimos N jogos de cada equipa (casa / fora) para os golos esperados
JOGOS_ALFA = 20         # jogos por equipa usados nas frequências históricas (alfa)  <-- PRESSUPOSTO, validar com a folha
MIN_JOGOS = 5           # mínimo de jogos de cada equipa para analisar
FATOR_CASA = 0.05       # vantagem de casa (5%)
MAX_GOLOS = 10

LIM_BTTS = 0.65         # BTTS >= 65%
LIM_RAROS = 0.12        # eventos raros ("2GV") >= 12%
EXIGIR_AMBOS = True     # True = BTTS E raros; False = qualquer um dos dois

# Resultados raros (casa, fora). Os dois últimos (4-2 e 2-4) são necessários para
# reproduzir o 18,49% do exemplo da folha; remove-os se não os quiseres.
RAROS = [(2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4), (4, 2), (2, 4)]
BAIXOS = [(0, 0), (1, 0), (0, 1), (1, 1)]

DIAS_A_FRENTE = 0       # 0 = só jogos de hoje; 1 = hoje e amanhã
ENVIAR_SE_VAZIO = True  # envia mensagem mesmo sem jogos que cumpram os critérios
URL_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
URL_LIGA = "https://www.football-data.co.uk/mmz4281/{epoca}/{liga}.csv"
# =====================================================================


def codigo_epoca(d, recuo=0):
    ano = (d.year if d.month >= 7 else d.year - 1) - recuo
    return f"{ano % 100:02d}{(ano + 1) % 100:02d}"


def ler_csv(url):
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    texto = r.content.decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(texto), on_bad_lines="skip")


def carregar_historico(liga, hoje):
    partes = []
    for recuo in (0, 1):  # época atual + anterior
        df = ler_csv(URL_LIGA.format(epoca=codigo_epoca(hoje, recuo), liga=liga))
        if df is not None:
            partes.append(df)
    if not partes:
        return None
    df = pd.concat(partes, ignore_index=True)
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df


def pmf(lam, n=MAX_GOLOS):
    return np.array([exp(-lam) * lam ** k / factorial(k) for k in range(n + 1)])


def golos_esperados(media_liga_casa, media_liga_fora, casa_marc, casa_sof, fora_marc, fora_sof):
    base_c = casa_marc * fora_sof / media_liga_casa
    base_f = fora_marc * casa_sof / media_liga_fora
    total = base_c + base_f
    lam_c = base_c * (1 + FATOR_CASA)
    lam_f = max(total - lam_c, 0.05)   # mantém o total de golos
    return lam_c, lam_f


def freq_historica(h, a):
    """Frequência de 0-0, 1-0, 0-1 e 1-1 nos jogos das duas equipas (casa da casa + fora do visitante)."""
    jogos = pd.concat([h.tail(JOGOS_ALFA)[["FTHG", "FTAG"]], a.tail(JOGOS_ALFA)[["FTHG", "FTAG"]]])
    n = len(jogos)
    piso = 0.5 / n  # evita alfa = 0 quando o resultado nunca ocorreu na amostra
    return {s: max(((jogos.FTHG == s[0]) & (jogos.FTAG == s[1])).sum() / n, piso) for s in BAIXOS}


def matriz(lam_c, lam_f, freq_baixos, normalizar=True):
    P = np.outer(pmf(lam_c), pmf(lam_f))          # linhas = casa, colunas = fora
    mascara = np.ones_like(P, dtype=bool)
    for s in BAIXOS:
        mascara[s] = False
    soma_baixos = sum(freq_baixos.values())
    resto = P[mascara].sum()
    beta = max((1 - soma_baixos) / resto, 0.0) if normalizar else 1.0
    M = P.copy()
    M[mascara] = P[mascara] * beta
    for s in BAIXOS:
        M[s] = freq_baixos[s]                      # = Poisson x alfa
    return M, beta


def metricas(M):
    btts = M[1:, 1:].sum()
    raros = sum(M[s] for s in RAROS)
    return btts, raros


def analisar(hist, casa, fora, hoje, normalizar=True):
    h = hist[(hist.HomeTeam == casa) & (hist.Date < hoje)]
    a = hist[(hist.AwayTeam == fora) & (hist.Date < hoje)]
    if len(h) < MIN_JOGOS or len(a) < MIN_JOGOS:
        return None
    h10, a10 = h.tail(JOGOS_AMOSTRA), a.tail(JOGOS_AMOSTRA)
    lam_c, lam_f = golos_esperados(
        hist.FTHG.mean(), hist.FTAG.mean(),
        h10.FTHG.mean(), h10.FTAG.mean(),
        a10.FTAG.mean(), a10.FTHG.mean(),
    )
    freq = freq_historica(h, a)
    M, beta = matriz(lam_c, lam_f, freq, normalizar)
    btts, raros = metricas(M)
    return {"lam_c": lam_c, "lam_f": lam_f, "beta": beta, "freq": freq,
            "btts": btts, "raros": raros, "soma": M.sum(),
            "liga": (hist.FTHG.mean(), hist.FTAG.mean(), len(hist), hist.Date.min(), hist.Date.max()),
            "equipas": (h10.FTHG.mean(), h10.FTAG.mean(), a10.FTAG.mean(), a10.FTHG.mean()),
            "h10": h10, "a10": a10}


def cumpre(r):
    ok_b, ok_r = r["btts"] >= LIM_BTTS, r["raros"] >= LIM_RAROS
    return (ok_b and ok_r) if EXIGIR_AMBOS else (ok_b or ok_r)


def odd(p):
    return 1 / p if p > 0 else float("inf")


def enviar_telegram(texto):
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise SystemExit("ERRO: TELEGRAM_TOKEN / TELEGRAM_CHAT_ID não definidos (verifica os segredos no GitHub).")
    bloco = ""
    for linha in texto.split("\n") + [None]:
        if linha is None or len(bloco) + len(linha) > 3800:
            if bloco:
                r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                  json={"chat_id": chat, "text": bloco}, timeout=30)
                if not r.ok:
                    raise SystemExit(f"ERRO do Telegram ({r.status_code}): {r.text}")
                print("Mensagem enviada para o Telegram.")
            bloco = ""
        if linha is not None:
            bloco += linha + "\n"


def simular(liga, casa, fora, hoje):
    hist = carregar_historico(liga, hoje)
    if hist is None:
        print("Não consegui descarregar os dados da liga.")
        return
    for norm in (False, True):
        r = analisar(hist, casa, fora, hoje, normalizar=norm)
        if r is None:
            print("Poucos jogos de uma das equipas.")
            return
        if not norm:
            lc, lf, n, d0, d1 = r["liga"]
            cm, cs, fm, fs = r["equipas"]
            print(f"\nLiga: {n} jogos ({d0:%d/%m/%Y} a {d1:%d/%m/%Y}) | média golos casa {lc:.2f}, fora {lf:.2f}")
            print(f"{casa} em casa (últimos {len(r['h10'])}): marcados {cm:.2f}, sofridos {cs:.2f}")
            print(r["h10"].assign(Date=r["h10"].Date.dt.strftime("%d/%m/%Y")).to_string(index=False, header=False))
            print(f"{fora} fora (últimos {len(r['a10'])}): marcados {fm:.2f}, sofridos {fs:.2f}")
            print(r["a10"].assign(Date=r["a10"].Date.dt.strftime("%d/%m/%Y")).to_string(index=False, header=False))
        print(f"\n--- {'NORMALIZADO (β automático)' if norm else 'SEM normalizar (β=1, como na folha)'} ---")
        print(f"Golos esperados: {r['lam_c']:.2f} - {r['lam_f']:.2f} | β={r['beta']:.3f} | soma da matriz={r['soma']:.2%}")
        print("Frequências 0-0/1-0/0-1/1-1:", {f"{k[0]}-{k[1]}": f"{v:.2%}" for k, v in r["freq"].items()})
        print(f"BTTS={r['btts']:.1%}  Raros={r['raros']:.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simular", nargs=3, metavar=("LIGA", "CASA", "FORA"))
    ap.add_argument("--sem-telegram", action="store_true")
    args = ap.parse_args()
    hoje = pd.Timestamp(date.today())

    if args.simular:
        simular(*args.simular, hoje)
        return

    fx = ler_csv(URL_FIXTURES)
    if fx is None:
        raise SystemExit("ERRO: não consegui descarregar o fixtures.csv")
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    fx = fx[(fx.Date >= hoje) & (fx.Date <= hoje + pd.Timedelta(days=DIAS_A_FRENTE))]
    fx = fx[fx.Div.isin(LIGAS)]

    alertas, analisados = [], 0
    for liga in fx.Div.unique():
        hist = carregar_historico(liga, hoje)
        if hist is None:
            continue
        for _, j in fx[fx.Div == liga].iterrows():
            r = analisar(hist, j.HomeTeam, j.AwayTeam, hoje)
            if r is None:
                continue
            analisados += 1
            if cumpre(r):
                alertas.append((liga, j, r))

    cab = f"⚽ Alertas {hoje:%d/%m/%Y} — {analisados} jogos analisados"
    if not alertas:
        texto = cab + "\nNenhum jogo cumpre os critérios hoje."
    else:
        linhas = [cab, f"Critérios: BTTS ≥ {LIM_BTTS:.0%} {'e' if EXIGIR_AMBOS else 'ou'} raros ≥ {LIM_RAROS:.0%}\n"]
        for liga, j, r in sorted(alertas, key=lambda x: -x[2]["raros"]):
            hora = j.get("Time", "")
            linhas.append(f"{LIGAS[liga]} {hora}\n{j.HomeTeam} vs {j.AwayTeam}\n"
                          f"BTTS {r['btts']:.1%} (odd {odd(r['btts']):.2f}) | "
                          f"Raros {r['raros']:.1%} (odd {odd(r['raros']):.2f})\n"
                          f"Golos esperados {r['lam_c']:.2f} - {r['lam_f']:.2f}\n")
        texto = "\n".join(linhas)

    print(texto)
    if not args.sem_telegram and (alertas or ENVIAR_SE_VAZIO):
        enviar_telegram(texto)


if __name__ == "__main__":
    main()
