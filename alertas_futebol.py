#!/usr/bin/env python3
"""
Alertas diários de futebol (BTTS + resultados raros) por Telegram.

Replica o "Modelo de Poisson simplificado corrigido" da folha Google Sheets:
  1. Golos esperados = (marcados x sofridos do adversário) / média da liga
  2. Ajuste de vantagem de casa (+5%), mantendo o total de golos
  3. Matriz de Poisson 0..10 x 0..10
  4. Células 0-0, 1-0, 0-1, 1-1 passam a ser a frequência histórica da liga
     (época atual + anterior) = Poisson x alfa
  5. As restantes células são multiplicadas por beta (β). Com NORMALIZAR = True, o β é
     calculado para a matriz somar 100%; com False, β = 1 (como a folha)

Uso:
  python alertas_futebol.py                          # corre os jogos de hoje e envia para o Telegram
  python alertas_futebol.py --sem-telegram           # só imprime
  python alertas_futebol.py --simular SP1 Barcelona Villarreal   # um jogo à escolha (para validar com a folha)
  python alertas_futebol.py --simular MLS "Inter Miami" "Columbus Crew"   # MLS e BRA (Brasileirão); nomes com espaços entre aspas

  python alertas_futebol.py --simular-euro "FC Barcelona" "Paris Saint-Germain FC"   # jogo europeu (nomes como na API)

Variáveis de ambiente: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
Opcional (competições europeias): FOOTBALL_DATA_API_KEY (chave gratuita de football-data.org)
"""
import argparse
import difflib
import io
import os
import time
import re
import unicodedata
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

# Ligas "extra" (MLS, Brasileirão): DESLIGADAS. Para ativar, troca por:
#   {"MLS": {"nome": "MLS", "ficheiro": "USA", "pais": "USA", "liga": "MLS"},
#    "BRA": {"nome": "Brasileirão Série A", "ficheiro": "BRA", "pais": "Brazil", "liga": "Serie A"}}
LIGAS_EXTRA = {}

# Competições europeias (experimental, precisa de FOOTBALL_DATA_API_KEY): DESLIGADAS.
# Para ativar, troca por {"CL": "Liga dos Campeões", "EL": "Liga Europa"}
COMPETICOES_EUROPEIAS = {}

JOGOS_AMOSTRA = 10      # últimos N jogos de cada equipa (casa / fora) para os golos esperados
MIN_JOGOS = 10          # mínimo de jogos de cada equipa para analisar
FATOR_CASA = 0.05       # vantagem de casa (5%)

# Equipas promovidas/despromovidas: se tiverem menos de MIN_JOGOS jogos na liga, completam-se os jogos
# em falta (até JOGOS_AMOSTRA) com os da divisão vizinha, ajustados pela diferença de golos entre as duas ligas.
# Equipas com histórico completo NÃO são afetadas. Põe False para desligar.
COMPLETAR_COM_DIVISAO_VIZINHA = True
MIN_JOGOS_LIGA = 4      # mínimo de jogos na liga atual (nesse campo) para poder completar
VIZINHAS = {"E0": ["E1"], "E1": ["E0", "E2"], "SP1": ["SP2"], "SP2": ["SP1"], "D1": ["D2"], "D2": ["D1"],
            "I1": ["I2"], "I2": ["I1"], "F1": ["F2"], "F2": ["F1"]}
MAX_GOLOS = 10

LIM_BTTS = 0.65         # BTTS >= 65%
LIM_RAROS = 0.13        # eventos raros ("2GV") >= 13%
EXIGIR_AMBOS = True     # True = BTTS E raros; False = qualquer um dos dois
NORMALIZAR = False      # False = números iguais aos da folha (β=1, matriz soma >100%); True = β automático (soma 100%)

# Resultados raros (casa, fora). Os dois últimos (4-2 e 2-4) são necessários para
# reproduzir o 18,49% do exemplo da folha; remove-os se não os quiseres.
RAROS = [(2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4), (4, 2), (2, 4)]
BAIXOS = [(0, 0), (1, 0), (0, 1), (1, 1)]

DIAS_A_FRENTE = 1       # 0 = só jogos de hoje; 1 = hoje e amanhã
ENVIAR_SE_VAZIO = True  # envia mensagem mesmo sem jogos que cumpram os critérios
URL_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
URL_LIGA = "https://www.football-data.co.uk/mmz4281/{epoca}/{liga}.csv"
URL_FIXTURES_EXTRA = "https://www.football-data.co.uk/new_league_fixtures.csv"
URL_LIGA_EXTRA = "https://www.football-data.co.uk/new/{ficheiro}.csv"
URL_API = "https://api.football-data.org/v4/competitions/{cod}/matches"
# =====================================================================


def codigo_epoca(d, recuo=0):
    ano = (d.year if d.month >= 7 else d.year - 1) - recuo
    return f"{ano % 100:02d}{(ano + 1) % 100:02d}"


def ler_csv(url, tentativas=4, timeout=45):
    """Descarrega um CSV com novas tentativas (o football-data.co.uk às vezes demora ou falha)."""
    for i in range(1, tentativas + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                texto = r.content.decode("utf-8-sig", errors="replace")
                return pd.read_csv(io.StringIO(texto), on_bad_lines="skip")
            if r.status_code == 404:        # o ficheiro não existe (por exemplo, época sem dados)
                return None
            print(f"AVISO: HTTP {r.status_code} em {url} (tentativa {i}/{tentativas})")
        except requests.RequestException as e:
            print(f"AVISO: {type(e).__name__} em {url} (tentativa {i}/{tentativas})")
        if i < tentativas:
            time.sleep(5 * i)
    return None


def _limpar(df):
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df


def carregar_historico(liga, hoje):
    partes = []
    for recuo in (0, 1):  # época atual + anterior
        df = ler_csv(URL_LIGA.format(epoca=codigo_epoca(hoje, recuo), liga=liga))
        if df is not None:
            partes.append(df)
    if not partes:
        return None
    return _limpar(pd.concat(partes, ignore_index=True))


def carregar_historico_extra(cfg, hoje):
    """MLS / Brasileirão: um único CSV com todas as épocas; usa o ano atual + o anterior."""
    df = ler_csv(URL_LIGA_EXTRA.format(ficheiro=cfg["ficheiro"]))
    if df is None:
        return None
    if "Country" in df.columns and "League" in df.columns:
        df = df[(df["Country"] == cfg["pais"]) & (df["League"] == cfg["liga"])]
    epoca = pd.to_numeric(df["Season"], errors="coerce")
    df = df[epoca >= hoje.year - 1]
    df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
    return _limpar(df)


_CACHE = {}


def carregar(liga, hoje):
    chave = (liga, hoje)
    if chave not in _CACHE:
        if liga in LIGAS_EXTRA:
            _CACHE[chave] = carregar_historico_extra(LIGAS_EXTRA[liga], hoje)
        else:
            _CACHE[chave] = carregar_historico(liga, hoje)
    return _CACHE[chave]


def nome_liga(liga):
    if liga in LIGAS_EXTRA:
        return LIGAS_EXTRA[liga]["nome"]
    if liga in COMPETICOES_EUROPEIAS:
        return COMPETICOES_EUROPEIAS[liga] + " (experimental)"
    return LIGAS[liga]


def pmf(lam, n=MAX_GOLOS):
    return np.array([exp(-lam) * lam ** k / factorial(k) for k in range(n + 1)])


def golos_esperados(media_liga_casa, media_liga_fora, casa_marc, casa_sof, fora_marc, fora_sof):
    base_c = casa_marc * fora_sof / media_liga_casa
    base_f = fora_marc * casa_sof / media_liga_fora
    total = base_c + base_f
    lam_c = base_c * (1 + FATOR_CASA)
    lam_f = max(total - lam_c, 0.05)   # mantém o total de golos
    return lam_c, lam_f


def freq_historica(hist):
    """Frequência de 0-0, 1-0, 0-1 e 1-1 em todos os jogos da liga (época atual + anterior)."""
    n = len(hist)
    return {s: ((hist.FTHG == s[0]) & (hist.FTAG == s[1])).sum() / n for s in BAIXOS}


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


def completar(hist, liga, equipa, em_casa, hoje, atuais):
    """Completa os jogos de uma equipa promovida/despromovida com os da divisão vizinha.
    Os golos são reescalados pela razão entre as médias das duas ligas (marcados e sofridos)."""
    faltam = JOGOS_AMOSTRA - len(atuais)
    if faltam <= 0 or len(atuais) < MIN_JOGOS_LIGA:
        return atuais, None
    for viz in VIZINHAS.get(liga, []):
        hv = carregar(viz, hoje)
        if hv is None:
            continue
        nomes = set(hv.HomeTeam) | set(hv.AwayTeam)
        nome_v = equipa if equipa in nomes else None
        if nome_v is None:
            mini = {n: (normalizar_nome(n), viz, hv.Date.max()) for n in nomes}
            achado = encontrar_equipa(equipa, mini)
            nome_v = achado[1] if achado else None
        if nome_v is None:
            continue
        col = "HomeTeam" if em_casa else "AwayTeam"
        extra = hv[(hv[col] == nome_v) & (hv.Date < hoje)].tail(faltam).copy()
        if extra.empty:
            continue
        extra["FTHG"] = extra.FTHG * (hist.FTHG.mean() / hv.FTHG.mean())
        extra["FTAG"] = extra.FTAG * (hist.FTAG.mean() / hv.FTAG.mean())
        nome = nome_liga(viz) if viz in LIGAS else viz
        return pd.concat([extra, atuais]), f"{equipa}: {len(atuais)} jogos na liga + {len(extra)} da {nome}"
    return atuais, None


def analisar(hist, casa, fora, hoje, normalizar=None, liga=None):
    if normalizar is None:
        normalizar = NORMALIZAR
    h = hist[(hist.HomeTeam == casa) & (hist.Date < hoje)]
    a = hist[(hist.AwayTeam == fora) & (hist.Date < hoje)]
    notas = []
    if COMPLETAR_COM_DIVISAO_VIZINHA and liga and (len(h) < MIN_JOGOS or len(a) < MIN_JOGOS):
        if len(h) < MIN_JOGOS:
            h, nota = completar(hist, liga, casa, True, hoje, h)
            notas += [nota] if nota else []
        if len(a) < MIN_JOGOS:
            a, nota = completar(hist, liga, fora, False, hoje, a)
            notas += [nota] if nota else []
    if len(h) < MIN_JOGOS or len(a) < MIN_JOGOS:
        return None
    h10, a10 = h.tail(JOGOS_AMOSTRA), a.tail(JOGOS_AMOSTRA)
    lam_c, lam_f = golos_esperados(
        hist.FTHG.mean(), hist.FTAG.mean(),
        h10.FTHG.mean(), h10.FTAG.mean(),
        a10.FTAG.mean(), a10.FTHG.mean(),
    )
    freq = freq_historica(hist)
    M, beta = matriz(lam_c, lam_f, freq, normalizar)
    btts, raros = metricas(M)
    return {"lam_c": lam_c, "lam_f": lam_f, "beta": beta, "freq": freq,
            "btts": btts, "raros": raros, "soma": M.sum(),
            "liga": (hist.FTHG.mean(), hist.FTAG.mean(), len(hist), hist.Date.min(), hist.Date.max()),
            "equipas": (h10.FTHG.mean(), h10.FTAG.mean(), a10.FTAG.mean(), a10.FTHG.mean()),
            "h10": h10, "a10": a10, "notas": notas}


# ------------------- competições europeias (experimental) -------------------
RUIDO = {"fc", "cf", "afc", "sc", "ac", "as", "ss", "ssc", "rc", "cd", "ud", "sv", "fk", "sk",
         "bsc", "vfb", "vfl", "tsg", "club", "clube", "de", "calcio", "fotbal"}
ALIAS = {  # nome da API (normalizado) -> nome no football-data.co.uk
    "manchester city": "Man City", "manchester united": "Man United",
    "paris saint germain": "Paris SG", "atletico madrid": "Ath Madrid",
    "athletic": "Ath Bilbao", "athletic bilbao": "Ath Bilbao",
    "internazionale milano": "Inter", "internazionale": "Inter", "inter milan": "Inter",
    "bayern munchen": "Bayern Munich", "borussia monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt", "sporting portugal": "Sp Lisbon", "sporting cp": "Sp Lisbon",
    "psv": "PSV Eindhoven", "wolverhampton wanderers": "Wolves", "nottingham forest": "Nott'm Forest",
    "olympique lyonnais": "Lyon", "stade brestois": "Brest",
    "royale union saint gilloise": "St. Gilloise", "union saint gilloise": "St. Gilloise",
}


def normalizar_nome(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
    palavras = [p for p in re.sub(r"[^a-z0-9 ]", " ", n).split() if p not in RUIDO and not p.isdigit()]
    return " ".join(palavras)


def construir_indice(hoje):
    """Equipas de todas as ligas nacionais: nome -> (nome normalizado, liga, data do último jogo)."""
    indice = {}
    for cod in LIGAS:
        hist = carregar(cod, hoje)
        if hist is None:
            continue
        for nome in set(hist.HomeTeam) | set(hist.AwayTeam):
            ultima = hist[(hist.HomeTeam == nome) | (hist.AwayTeam == nome)].Date.max()
            if nome not in indice or ultima > indice[nome][2]:   # equipa promovida/despromovida: fica a liga atual
                indice[nome] = (normalizar_nome(nome), cod, ultima)
    return indice


def encontrar_equipa(nome_api, indice):
    n = normalizar_nome(nome_api)
    alvo = ALIAS.get(n)
    if alvo and alvo in indice:
        return indice[alvo][1], alvo
    for nome, (nn, cod, _) in indice.items():
        if nn == n:
            return cod, nome
    cand = []
    for nome, (nn, cod, _) in indice.items():
        if min(len(nn), len(n)) >= 4 and (f" {nn} " in f" {n} " or f" {n} " in f" {nn} "):
            cand.append((len(nn), nome, cod))
    if cand:
        _, nome, cod = max(cand)
        return cod, nome
    melhor = difflib.get_close_matches(n, [v[0] for v in indice.values()], n=1, cutoff=0.85)
    if melhor:
        for nome, (nn, cod, _) in indice.items():
            if nn == melhor[0]:
                return cod, nome
    return None


def analisar_europeu(hoje, casa, fora, indice, normalizar=None):
    """Jogo entre equipas de ligas diferentes: força de cada equipa relativa à média da sua liga,
    aplicada à média de referência dos dois campeonatos."""
    if normalizar is None:
        normalizar = NORMALIZAR
    ec, ef = encontrar_equipa(casa, indice), encontrar_equipa(fora, indice)
    if ec is None or ef is None:
        faltam = [n for n, e in ((casa, ec), (fora, ef)) if e is None]
        print("AVISO: sem dados para: " + ", ".join(faltam))
        return None
    (lig_c, nome_c), (lig_f, nome_f) = ec, ef
    hc, hf = carregar(lig_c, hoje), carregar(lig_f, hoje)
    h = hc[(hc.HomeTeam == nome_c) & (hc.Date < hoje)]
    a = hf[(hf.AwayTeam == nome_f) & (hf.Date < hoje)]
    if len(h) < MIN_JOGOS or len(a) < MIN_JOGOS:
        print(f"AVISO: poucos jogos para {casa} ({len(h)}) ou {fora} ({len(a)})")
        return None
    h10, a10 = h.tail(JOGOS_AMOSTRA), a.tail(JOGOS_AMOSTRA)
    lh_c, la_c = hc.FTHG.mean(), hc.FTAG.mean()      # médias da liga da casa (golos casa / golos fora)
    lh_f, la_f = hf.FTHG.mean(), hf.FTAG.mean()      # médias da liga do visitante
    ref_h, ref_a = (lh_c + lh_f) / 2, (la_c + la_f) / 2
    base_c = (h10.FTHG.mean() / lh_c) * (a10.FTHG.mean() / lh_f) * ref_h
    base_f = (a10.FTAG.mean() / la_f) * (h10.FTAG.mean() / la_c) * ref_a
    lam_c = base_c * (1 + FATOR_CASA)
    lam_f = max(base_c + base_f - lam_c, 0.05)
    fc, ff = freq_historica(hc), freq_historica(hf)
    freq = {s: (fc[s] + ff[s]) / 2 for s in BAIXOS}
    M, beta = matriz(lam_c, lam_f, freq, normalizar)
    btts, raros = metricas(M)
    return {"lam_c": lam_c, "lam_f": lam_f, "beta": beta, "freq": freq, "btts": btts, "raros": raros,
            "soma": M.sum(), "equipas_csv": (nome_c, lig_c, nome_f, lig_f)}


def fixtures_europeus(chave, hoje):
    jogos = []
    ate = hoje + pd.Timedelta(days=DIAS_A_FRENTE)
    for cod, nome in COMPETICOES_EUROPEIAS.items():
        try:
            r = requests.get(URL_API.format(cod=cod), headers={"X-Auth-Token": chave},
                             params={"dateFrom": f"{hoje:%Y-%m-%d}", "dateTo": f"{ate:%Y-%m-%d}"}, timeout=30)
        except requests.RequestException as e:
            print(f"AVISO: {nome} indisponível ({e})")
            continue
        if r.status_code != 200:
            print(f"AVISO: {nome} ({cod}) indisponível: HTTP {r.status_code}")
            continue
        for mj in r.json().get("matches", []):
            if mj.get("status") not in ("SCHEDULED", "TIMED"):
                continue
            c, f = (mj.get("homeTeam") or {}).get("name"), (mj.get("awayTeam") or {}).get("name")
            if c and f:
                jogos.append((cod, {"HomeTeam": c, "AwayTeam": f, "Time": mj.get("utcDate", "")[11:16] + " UTC"}))
    return jogos


def motivo_ignorado(hist, casa, fora, hoje):
    """Explica porque um jogo não foi analisado (histórico insuficiente ou nome diferente)."""
    h = hist[(hist.HomeTeam == casa) & (hist.Date < hoje)]
    a = hist[(hist.AwayTeam == fora) & (hist.Date < hoje)]
    partes = []
    if len(h) < MIN_JOGOS:
        partes.append(f"{casa}: {len(h)} jogos em casa" if len(h) else f"{casa}: sem jogos no histórico (nome diferente?)")
    if len(a) < MIN_JOGOS:
        partes.append(f"{fora}: {len(a)} jogos fora" if len(a) else f"{fora}: sem jogos no histórico (nome diferente?)")
    return "; ".join(partes) or "sem dados suficientes"


DIAS_SEMANA = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]


def nome_dia(d, hoje, curto=False):
    """HOJE / AMANHÃ + dia da semana e data. curto=True: 'HOJE 27/11'."""
    rel = {0: "HOJE", 1: "AMANHÃ"}.get((d.normalize() - hoje).days)
    if curto:
        return f"{rel} {d:%d/%m}" if rel else f"{d:%d/%m}"
    base = f"{DIAS_SEMANA[d.weekday()]} {d:%d/%m}"
    return f"{rel} ({base})" if rel else base


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
    hist = carregar(liga, hoje)
    if hist is None:
        print("Não consegui descarregar os dados da liga.")
        return
    for norm in (False, True):
        r = analisar(hist, casa, fora, hoje, normalizar=norm, liga=liga)
        if r is None:
            print(f"Jogo ignorado: {motivo_ignorado(hist, casa, fora, hoje)} (mínimo {MIN_JOGOS} jogos).")
            return
        if not norm:
            for nota in r["notas"]:
                print(f"HISTÓRICO COMPLETADO -> {nota} (golos reescalados)")
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


def simular_euro(casa, fora, hoje):
    indice = construir_indice(hoje)
    for norm in (False, True):
        r = analisar_europeu(hoje, casa, fora, indice, normalizar=norm)
        if r is None:
            return
        nc, lc, nf, lf = r["equipas_csv"]
        if not norm:
            print(f"\n{casa} -> {nc} ({lc}) | {fora} -> {nf} ({lf})")
        print(f"\n--- {'NORMALIZADO (β automático)' if norm else 'SEM normalizar (β=1, como na folha)'} ---")
        print(f"Golos esperados: {r['lam_c']:.2f} - {r['lam_f']:.2f} | β={r['beta']:.3f} | soma={r['soma']:.2%}")
        print(f"BTTS={r['btts']:.1%}  Raros={r['raros']:.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simular", nargs=3, metavar=("LIGA", "CASA", "FORA"))
    ap.add_argument("--simular-euro", nargs=2, metavar=("CASA", "FORA"))
    ap.add_argument("--sem-telegram", action="store_true")
    args = ap.parse_args()
    hoje = pd.Timestamp(date.today())

    if args.simular:
        simular(*args.simular, hoje)
        return
    if args.simular_euro:
        simular_euro(*args.simular_euro, hoje)
        return

    fx = ler_csv(URL_FIXTURES)
    if fx is None:
        raise SystemExit("ERRO: não consegui descarregar o fixtures.csv")
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    datas_fx = fx[fx.Div.isin(LIGAS)].Date.dropna()      # datas que o ficheiro de jogos contém
    if len(datas_fx):
        print(f"fixtures.csv: jogos de {datas_fx.min():%d/%m/%Y} a {datas_fx.max():%d/%m/%Y}")
    fx = fx[(fx.Date >= hoje) & (fx.Date <= hoje + pd.Timedelta(days=DIAS_A_FRENTE))]
    fx = fx[fx.Div.isin(LIGAS)]

    jogos_por_liga = {liga: fx[fx.Div == liga] for liga in fx.Div.unique()}

    # ligas extra (MLS, Brasileirão)
    fxe = ler_csv(URL_FIXTURES_EXTRA) if LIGAS_EXTRA else None
    if LIGAS_EXTRA and fxe is None:
        print("AVISO: não consegui descarregar o new_league_fixtures.csv (MLS/Brasileirão ignoradas).")
    elif fxe is not None:
        fxe = fxe.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})
        fxe["Date"] = pd.to_datetime(fxe["Date"], dayfirst=True, errors="coerce")
        fxe = fxe[(fxe.Date >= hoje) & (fxe.Date <= hoje + pd.Timedelta(days=DIAS_A_FRENTE))]
        for cod, cfg in LIGAS_EXTRA.items():
            sub = fxe[(fxe.Country == cfg["pais"]) & (fxe.League == cfg["liga"])]
            if len(sub):
                jogos_por_liga[cod] = sub

    alertas, analisados, por_dia, ignorados = [], 0, {}, []
    for liga, jogos in jogos_por_liga.items():
        hist = carregar(liga, hoje)
        if hist is None:
            print(f"AVISO: sem histórico para {liga}; jogos ignorados.")
            continue
        for _, j in jogos.iterrows():
            r = analisar(hist, j.HomeTeam, j.AwayTeam, hoje, liga=liga)
            if r is None:
                ignorados.append((j, motivo_ignorado(hist, j.HomeTeam, j.AwayTeam, hoje)))
                continue
            analisados += 1
            d = j.get("Date")
            if pd.notna(d):
                por_dia[d.normalize()] = por_dia.get(d.normalize(), 0) + 1
            if cumpre(r):
                alertas.append((liga, j, r))

    # competições europeias (precisa da chave da API; se não existir, é ignorada)
    chave = os.environ.get("FOOTBALL_DATA_API_KEY")
    if chave:
        europeus = fixtures_europeus(chave, hoje)
        if europeus:
            indice = construir_indice(hoje)
            for cod, j in europeus:
                r = analisar_europeu(hoje, j["HomeTeam"], j["AwayTeam"], indice)
                if r is None:
                    continue
                analisados += 1
                if cumpre(r):
                    alertas.append((cod, pd.Series(j), r))

    cab = f"⚽ Alertas {hoje:%d/%m/%Y} — {analisados} jogos analisados"
    if por_dia:
        cab += " (" + ", ".join(f"{ {0: 'hoje', 1: 'amanhã'}.get((d - hoje).days, f'{d:%d/%m}') } {n}" for d, n in sorted(por_dia.items())) + ")"
    if analisados == 0:
        texto = cab + "\nNenhum jogo analisado: o ficheiro de jogos ainda não tem jogos para estes dias (ou as equipas não têm 10 jogos de histórico)."
    elif not alertas:
        texto = cab + "\nNenhum jogo cumpre os critérios."
    else:
        linhas = [cab, f"Critérios: BTTS ≥ {LIM_BTTS:.0%} {'e' if EXIGIR_AMBOS else 'ou'} raros ≥ {LIM_RAROS:.0%}"]
        dias = sorted({j.get("Date") for _, j, _ in alertas if pd.notna(j.get("Date"))})
        grupos = [(d, [a for a in alertas if a[1].get("Date") == d]) for d in dias]
        grupos += [(None, [a for a in alertas if pd.isna(a[1].get("Date"))])]
        for d, lista in grupos:
            if not lista:
                continue
            linhas.append(f"\n📅 {nome_dia(d, hoje) if d is not None else 'Data desconhecida'} — {len(lista)} alerta(s)")
            for liga, j, r in sorted(lista, key=lambda x: (str(x[1].get("Time", "")) or "99:99", -x[2]["raros"])):
                etiqueta = f"{nome_dia(d, hoje, curto=True)} " if d is not None else ""
                linhas.append(f"\n{nome_liga(liga)} — {etiqueta}{j.get('Time', '')}\n{j.HomeTeam} vs {j.AwayTeam}\n"
                              f"BTTS {r['btts']:.1%} (odd {odd(r['btts']):.2f}) | "
                              f"Raros {r['raros']:.1%} (odd {odd(r['raros']):.2f})\n"
                              f"Golos esperados {r['lam_c']:.2f} - {r['lam_f']:.2f}"
                              + (f"\n⚠️ Histórico completado: {'; '.join(r['notas'])}" if r.get("notas") else ""))
        texto = "\n".join(linhas)

    for j, motivo in ignorados:      # só no registo do workflow; não vai para o alerta
        print(f"IGNORADO: {j.HomeTeam} vs {j.AwayTeam} ({motivo})")

    if len(datas_fx):
        texto += f"\n\nℹ️ O ficheiro de jogos tem datas de {datas_fx.min():%d/%m} a {datas_fx.max():%d/%m}."

    print(texto)
    if not args.sem_telegram and (alertas or ENVIAR_SE_VAZIO):
        enviar_telegram(texto)


if __name__ == "__main__":
    main()
