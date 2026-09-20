
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings
import html

from datetime import datetime
from zoneinfo import ZoneInfo
from io import StringIO

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

NIFTY_200_URL = (
    "https://www.niftyindices.com/"
    "IndexConstituent/ind_nifty200list.csv"
)

VAL_PROXIMITY_PCT = 2.0

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

DATA_PERIOD = "1y"

PIVOT_WINDOW = 3

VALUE_AREA_PERCENT = 70

VP_BINS = 50


# ============================================================
# 1. GET NIFTY 200
# ============================================================

def get_nifty200():

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*"
    }

    response = requests.get(
        NIFTY_200_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(
        StringIO(response.text)
    )

    df.columns = df.columns.str.strip()

    df = df[
        ["Company Name", "Symbol"]
    ].copy()

    df["Yahoo Symbol"] = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        + ".NS"
    )

    return df


# ============================================================
# 2. DOWNLOAD MARKET DATA
# ============================================================

def download_market_data(nifty200):

    market_data = {}

    symbols = nifty200[
        "Yahoo Symbol"
    ].tolist()

    print("Downloading daily market data...")

    for i, symbol in enumerate(symbols, 1):

        try:

            df = yf.download(
                symbol,
                period=DATA_PERIOD,
                interval="1d",
                auto_adjust=False,
                progress=False
            )

            if df is None or df.empty:
                continue

            if isinstance(
                df.columns,
                pd.MultiIndex
            ):
                df.columns = (
                    df.columns
                    .get_level_values(0)
                )

            df = df[
                [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]
            ].copy()

            df.dropna(inplace=True)

            if len(df) < 50:
                continue

            market_data[symbol] = df

            if i % 20 == 0:
                print(
                    f"Downloaded: "
                    f"{i}/{len(symbols)}"
                )

        except Exception:
            continue

    print(
        "Successful stocks:",
        len(market_data)
    )

    return market_data


# ============================================================
# 3. FIND SWING LOW → SWING HIGH
# ============================================================

def find_recent_swing_period(
    df,
    window=PIVOT_WINDOW
):

    data = df.copy()

    highs = data["High"].values
    lows = data["Low"].values

    swing_highs = []
    swing_lows = []

    for i in range(
        window,
        len(data) - window
    ):

        left_highs = highs[
            i-window:i
        ]

        right_highs = highs[
            i+1:i+window+1
        ]

        if (
            highs[i] >= max(left_highs)
            and
            highs[i] >= max(right_highs)
        ):
            swing_highs.append(i)

        left_lows = lows[
            i-window:i
        ]

        right_lows = lows[
            i+1:i+window+1
        ]

        if (
            lows[i] <= min(left_lows)
            and
            lows[i] <= min(right_lows)
        ):
            swing_lows.append(i)

    if not swing_highs:
        return None

    high_idx = swing_highs[-1]

    valid_lows = [
        idx
        for idx in swing_lows
        if idx < high_idx
    ]

    if not valid_lows:
        return None

    low_idx = valid_lows[-1]

    return {

        "swing_low_idx": low_idx,

        "swing_high_idx": high_idx,

        "swing_low_date":
            data.index[low_idx],

        "swing_high_date":
            data.index[high_idx],

        "swing_low_price":
            float(data.iloc[low_idx]["Low"]),

        "swing_high_price":
            float(data.iloc[high_idx]["High"])
    }


# ============================================================
# 4. VOLUME PROFILE
# ============================================================

def calculate_volume_profile(
    df,
    start_idx,
    end_idx,
    bins=VP_BINS
):

    data = df.iloc[
        start_idx:end_idx + 1
    ].copy()

    if data.empty:
        return None

    price_low = float(
        data["Low"].min()
    )

    price_high = float(
        data["High"].max()
    )

    if price_high <= price_low:
        return None

    edges = np.linspace(
        price_low,
        price_high,
        bins + 1
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2

    volume_profile = np.zeros(
        bins
    )

    for _, row in data.iterrows():

        day_low = float(row["Low"])
        day_high = float(row["High"])
        day_volume = float(row["Volume"])

        if day_high <= day_low:
            continue

        touched_bins = np.where(
            (centers >= day_low)
            &
            (centers <= day_high)
        )[0]

        if len(touched_bins) == 0:
            continue

        volume_per_bin = (
            day_volume
            / len(touched_bins)
        )

        volume_profile[
            touched_bins
        ] += volume_per_bin

    poc_idx = int(
        np.argmax(volume_profile)
    )

    poc = float(
        centers[poc_idx]
    )

    total_volume = (
        volume_profile.sum()
    )

    if total_volume <= 0:
        return None

    target_volume = (
        total_volume
        * VALUE_AREA_PERCENT
        / 100
    )

    included = {poc_idx}

    accumulated_volume = (
        volume_profile[poc_idx]
    )

    left = poc_idx - 1
    right = poc_idx + 1

    while accumulated_volume < target_volume:

        left_volume = (
            volume_profile[left]
            if left >= 0
            else -1
        )

        right_volume = (
            volume_profile[right]
            if right < bins
            else -1
        )

        if right_volume >= left_volume:

            if right < bins:

                included.add(right)

                accumulated_volume += (
                    right_volume
                )

                right += 1

            else:
                break

        else:

            if left >= 0:

                included.add(left)

                accumulated_volume += (
                    left_volume
                )

                left -= 1

            else:
                break

    val_idx = min(included)
    vah_idx = max(included)

    val = float(
        edges[val_idx]
    )

    vah = float(
        edges[vah_idx + 1]
    )

    return {

        "VAL": val,

        "POC": poc,

        "VAH": vah
    }


# ============================================================
# 5. MACD
# ============================================================

def calculate_macd(df):

    data = df.copy()

    ema_fast = (
        data["Close"]
        .ewm(
            span=MACD_FAST,
            adjust=False
        )
        .mean()
    )

    ema_slow = (
        data["Close"]
        .ewm(
            span=MACD_SLOW,
            adjust=False
        )
        .mean()
    )

    data["MACD"] = (
        ema_fast
        - ema_slow
    )

    data["Signal"] = (
        data["MACD"]
        .ewm(
            span=MACD_SIGNAL,
            adjust=False
        )
        .mean()
    )

    data["Histogram"] = (
        data["MACD"]
        - data["Signal"]
    )

    data["Bullish_Crossover"] = (

        (data["MACD"] > data["Signal"])

        &

        (
            data["MACD"].shift(1)
            <=
            data["Signal"].shift(1)
        )
    )

    crossover_dates = data.index[
        data["Bullish_Crossover"]
    ]

    current_macd = float(
        data["MACD"].iloc[-1]
    )

    current_signal = float(
        data["Signal"].iloc[-1]
    )

    current_histogram = float(
        data["Histogram"].iloc[-1]
    )

    zero_status = (
        "Above 0"
        if current_macd > 0
        else "Below 0"
    )

    direction = (
        "Rising"
        if current_macd
        >
        data["MACD"].iloc[-2]
        else "Falling"
    )

    if len(crossover_dates) == 0:

        return {

            "MACD": current_macd,

            "Signal": current_signal,

            "Histogram":
                current_histogram,

            "Crossover Date": None,

            "Crossover Age": None,

            "MACD Zero":
                zero_status,

            "MACD Direction":
                direction
        }

    crossover_date = (
        crossover_dates[-1]
    )

    crossover_position = (
        data.index.get_loc(
            crossover_date
        )
    )

    crossover_age = (
        len(data)
        - 1
        - crossover_position
    )

    return {

        "MACD": current_macd,

        "Signal": current_signal,

        "Histogram":
            current_histogram,

        "Crossover Date":
            crossover_date,

        "Crossover Age":
            crossover_age,

        "MACD Zero":
            zero_status,

        "MACD Direction":
            direction
    }


# ============================================================
# 6. RUN SCANNER
# ============================================================

def run_scanner():

    print()
    print("==============================================")
    print(" AURORA VOLUME PROFILE + MACD SCANNER")
    print("==============================================")
    print()

    nifty200 = get_nifty200()

    print(
        "Nifty 200 stocks:",
        len(nifty200)
    )

    market_data = (
        download_market_data(
            nifty200
        )
    )

    company_map = dict(
        zip(
            nifty200["Yahoo Symbol"],
            nifty200["Company Name"]
        )
    )

    results = []

    for symbol, df in market_data.items():

        swing = (
            find_recent_swing_period(df)
        )

        if swing is None:
            continue

        low_idx = swing[
            "swing_low_idx"
        ]

        high_idx = swing[
            "swing_high_idx"
        ]

        start_idx = min(
            low_idx,
            high_idx
        )

        end_idx = max(
            low_idx,
            high_idx
        )

        vp = calculate_volume_profile(
            df,
            start_idx,
            end_idx
        )

        if vp is None:
            continue

        macd = calculate_macd(df)

        current_price = float(
            df["Close"].iloc[-1]
        )

        val = float(vp["VAL"])

        poc = float(vp["POC"])

        vah = float(vp["VAH"])

        distance_from_val_pct = (
            (current_price - val)
            / val
        ) * 100

        # ----------------------------------------------------
        # PRICE CONDITION
        #
        # AT VAL OR UP TO 2% BELOW VAL
        # ----------------------------------------------------

        near_val = (

            current_price <= val

            and

            current_price
            >=
            val * (
                1
                -
                VAL_PROXIMITY_PCT / 100
            )
        )

        # ----------------------------------------------------
        # MACD CONDITIONS
        # ----------------------------------------------------

        bullish_crossover = (
            macd["Crossover Date"]
            is not None
        )

        macd_rising = (
            macd["MACD Direction"]
            == "Rising"
        )

        # ----------------------------------------------------
        # RVOL
        #
        # INFORMATION ONLY
        # NOT A FILTER
        # ----------------------------------------------------

        volume_ema_20 = (
            df["Volume"]
            .ewm(
                span=20,
                adjust=False
            )
            .mean()
        )

        current_volume = float(
            df["Volume"].iloc[-1]
        )

        current_volume_ema20 = float(
            volume_ema_20.iloc[-1]
        )

        if current_volume_ema20 > 0:

            rvol = (
                current_volume
                /
                current_volume_ema20
            )

        else:

            rvol = np.nan

        # ----------------------------------------------------
        # FINAL CANDIDATE
        # ----------------------------------------------------

        candidate = (

            near_val

            and

            bullish_crossover

            and

            macd_rising
        )

        results.append({

            "Symbol": symbol,

            "Company":
                company_map.get(
                    symbol,
                    symbol
                ),

            "Price":
                current_price,

            "VAL":
                val,

            "POC":
                poc,

            "VAH":
                vah,

            "Distance from VAL %":
                distance_from_val_pct,

            "VP Start":
                swing[
                    "swing_low_date"
                ],

            "VP End":
                swing[
                    "swing_high_date"
                ],

            "VP Sessions":
                (
                    high_idx
                    - low_idx
                    + 1
                ),

            "Latest Market Date":
                df.index[-1],

            "MACD":
                macd["MACD"],

            "Signal":
                macd["Signal"],

            "Histogram":
                macd["Histogram"],

            "MACD Crossover":
                macd["Crossover Date"],

            "Crossover Age":
                macd["Crossover Age"],

            "MACD Zero":
                macd["MACD Zero"],

            "MACD Direction":
                macd["MACD Direction"],

            "Latest Volume":
                current_volume,

            "Volume EMA 20":
                current_volume_ema20,

            "RVOL":
                rvol,

            "Candidate":
                candidate
        })

    scanner_df = pd.DataFrame(
        results
    )

    if scanner_df.empty:
        print("No data generated.")
        return scanner_df

    scanner_df = (
        scanner_df
        .sort_values(
            "Distance from VAL %",
            key=lambda x: abs(x)
        )
        .reset_index(drop=True)
    )

    candidates_df = (
        scanner_df[
            scanner_df["Candidate"]
            == True
        ]
        .copy()
    )

    print()
    print(
        "Stocks scanned:",
        len(scanner_df)
    )

    print(
        "Candidates:",
        len(candidates_df)
    )

    return scanner_df


# ============================================================
# 7. FORMAT HTML
# ============================================================

def fmt_date(value):

    if pd.isna(value):
        return "-"

    return pd.Timestamp(
        value
    ).strftime(
        "%d-%b-%Y"
    )


def fmt_num(value, decimals=2):

    if pd.isna(value):
        return "-"

    return f"{float(value):.{decimals}f}"


def fmt_rvol(value):

    if pd.isna(value):
        return "-"

    return (
        f"{float(value):.2f}×"
    )


def create_table(df):

    rows = []

    for _, row in df.iterrows():

        distance = (
            fmt_num(
                row[
                    "Distance from VAL %"
                ]
            )
            + "%"
        )

        if row["Candidate"]:
            row_class = (
                "candidate-row"
            )
        else:
            row_class = ""

        rows.append(f"""
<tr class="{row_class}">

<td class="stock">
{html.escape(
    str(row["Symbol"])
    .replace(".NS", "")
)}
</td>

<td>
{html.escape(
    str(row["Company"])
)}
</td>

<td>
{fmt_num(row["Price"])}
</td>

<td>
{fmt_num(row["VAL"])}
</td>

<td>
{fmt_num(row["POC"])}
</td>

<td>
{fmt_num(row["VAH"])}
</td>

<td>
{distance}
</td>

<td>
{fmt_date(row["VP Start"])}
</td>

<td>
{fmt_date(row["VP End"])}
</td>

<td>
{fmt_num(row["MACD"], 3)}
</td>

<td>
{fmt_num(row["Signal"], 3)}
</td>

<td>
{fmt_date(
    row["MACD Crossover"]
)}
</td>

<td>
{
    "-"
    if pd.isna(
        row["Crossover Age"]
    )
    else int(
        row["Crossover Age"]
    )
}
</td>

<td>
{html.escape(
    str(row["MACD Zero"])
)}
</td>

<td>
{html.escape(
    str(row["MACD Direction"])
)}
</td>

<td>
{fmt_rvol(row["RVOL"])}
</td>

</tr>
""")

    return "\n".join(rows)


# ============================================================
# 8. GENERATE WEBPAGE
# ============================================================

def generate_webpage(
    scanner_df
):

    candidates_df = (
        scanner_df[
            scanner_df["Candidate"]
            == True
        ]
        .copy()
    )

    refresh_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    refresh_display = (
        refresh_time.strftime(
            "%d-%b-%Y "
            "%I:%M:%S %p IST"
        )
    )

    # --------------------------------------------------------
    # Latest actual market-data date
    # --------------------------------------------------------

    latest_market_date = (
        pd.to_datetime(
            scanner_df[
                "Latest Market Date"
            ],
            errors="coerce"
        ).max()
    )

    latest_market_date_text = (
        latest_market_date.strftime(
            "%d-%b-%Y"
        )
        if pd.notna(latest_market_date)
        else "-"
    )

    candidate_table = create_table(
        candidates_df
    )

    all_table = create_table(
        scanner_df
    )

    html_content = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>
Aurora Volume Profile + MACD Scanner
</title>

<style>

* {{
box-sizing:border-box;
}}

body {{
margin:0;
font-family:Arial,Helvetica,sans-serif;
background:#f5f7fa;
color:#222;
}}

.header {{
background:#172033;
color:white;
padding:22px 28px;
}}

.header h1 {{
margin:0 0 8px 0;
font-size:26px;
}}

.header p {{
margin:4px 0;
color:#d8deea;
font-size:14px;
}}

.container {{
padding:20px;
max-width:1800px;
margin:auto;
}}

.cards {{
display:flex;
gap:14px;
flex-wrap:wrap;
margin-bottom:20px;
}}

.card {{
background:white;
padding:16px 20px;
border-radius:8px;
min-width:180px;
box-shadow:
0 1px 4px rgba(0,0,0,0.08);
}}

.card .label {{
font-size:12px;
color:#687386;
margin-bottom:6px;
}}

.card .value {{
font-size:22px;
font-weight:bold;
}}

.controls {{
background:white;
padding:15px;
border-radius:8px;
margin-bottom:15px;
box-shadow:
0 1px 4px rgba(0,0,0,0.08);
}}

input {{
width:100%;
max-width:420px;
padding:10px 12px;
border:1px solid #ccd3dd;
border-radius:6px;
font-size:14px;
}}

.tabs {{
margin-top:12px;
}}

button {{
padding:9px 16px;
border:none;
border-radius:5px;
margin-right:6px;
cursor:pointer;
background:#e6eaf0;
}}

button.active {{
background:#172033;
color:white;
}}

.table-wrapper {{
overflow-x:auto;
background:white;
border-radius:8px;
box-shadow:
0 1px 4px rgba(0,0,0,0.08);
}}

table {{
width:100%;
border-collapse:collapse;
font-size:13px;
white-space:nowrap;
}}

th {{
background:#eef1f5;
padding:11px 9px;
text-align:left;
}}

td {{
padding:9px;
border-bottom:
1px solid #edf0f4;
}}

tr:hover {{
background:#f7f9fc;
}}

.stock {{
font-weight:bold;
}}

.footer {{
margin-top:18px;
font-size:12px;
color:#6c7480;
}}

</style>

</head>

<body>

<div class="header">

<h1>
Aurora Volume Profile + MACD Scanner
</h1>

<p>
Nifty 200 | Daily Scanner
</p>

<p>
Last Refresh:
<strong>
{refresh_display}
</strong>
</p>

<p>
Latest Market Data:
<strong>
{latest_market_date_text}
</strong>
</p>

</div>

<div class="container">

<div class="cards">

<div class="card">
<div class="label">
Nifty 200 Scanned
</div>
<div class="value">
{len(scanner_df)}
</div>
</div>

<div class="card">
<div class="label">
Candidates
</div>
<div class="value">
{len(candidates_df)}
</div>
</div>

<div class="card">
<div class="label">
VAL Zone
</div>
<div class="value">
0% to -2%
</div>
</div>

<div class="card">
<div class="label">
MACD
</div>
<div class="value">
12 / 26 / 9
</div>
</div>

</div>

<div class="controls">

<input
type="text"
id="searchBox"
placeholder="Search stock or company..."
onkeyup="filterTable()"
>

<div class="tabs">

<button
id="candidateButton"
class="active"
onclick="showCandidates()">
Candidates
</button>

<button
id="allButton"
onclick="showAll()">
All Nifty 200
</button>

</div>

</div>

<div class="table-wrapper">

<table id="scannerTable">

<thead>

<tr>

<th>Stock</th>
<th>Company</th>
<th>Price</th>
<th>VAL</th>
<th>POC</th>
<th>VAH</th>
<th>Below VAL</th>
<th>VP Start</th>
<th>VP End</th>
<th>MACD</th>
<th>Signal</th>
<th>Cross Date</th>
<th>Cross Age</th>
<th>Zero</th>
<th>Direction</th>
<th>RVOL</th>

</tr>

</thead>

<tbody id="candidateBody">

{candidate_table}

</tbody>

<tbody
id="allBody"
style="display:none;">

{all_table}

</tbody>

</table>

</div>

<div class="footer">

Candidate conditions:

Price at VAL or up to 2% below VAL +

bullish MACD crossover +

MACD currently rising.

<br><br>

RVOL = Latest Daily Volume /

20-Day EMA Volume.

RVOL is displayed for information only.

</div>

</div>

<script>

function showCandidates() {{

document.getElementById(
"candidateBody"
).style.display="";

document.getElementById(
"allBody"
).style.display="none";

document.getElementById(
"candidateButton"
).classList.add("active");

document.getElementById(
"allButton"
).classList.remove("active");

filterTable();

}}

function showAll() {{

document.getElementById(
"candidateBody"
).style.display="none";

document.getElementById(
"allBody"
).style.display="";

document.getElementById(
"candidateButton"
).classList.remove("active");

document.getElementById(
"allButton"
).classList.add("active");

filterTable();

}}

function filterTable() {{

let input =
document.getElementById(
"searchBox"
).value.toLowerCase();

let activeBody;

if (
document.getElementById(
"candidateBody"
).style.display
!== "none"
) {{

activeBody =
document.getElementById(
"candidateBody"
);

}} else {{

activeBody =
document.getElementById(
"allBody"
);

}}

let rows =
activeBody.getElementsByTagName(
"tr"
);

for (
let i=0;
i<rows.length;
i++
) {{

let text =
rows[i].innerText.toLowerCase();

rows[i].style.display =
text.includes(input)
? ""
: "none";

}}

}}

</script>

</body>

</html>
"""

    with open(
        "index.html",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html_content)

    print()
    print(
        "index.html generated successfully."
    )

    print(
        "Last refresh:",
        refresh_display
    )


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    scanner_df = run_scanner()

    if not scanner_df.empty:

        scanner_df.to_csv(
            "volume_profile_macd_scanner_all.csv",
            index=False
        )

        scanner_df[
            scanner_df["Candidate"]
            == True
        ].to_csv(
            "volume_profile_macd_candidates.csv",
            index=False
        )

        generate_webpage(
            scanner_df
        )

        print()
        print(
            "Scanner completed successfully."
        )
