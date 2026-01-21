from pathlib import Path
import time
from typing import Iterable

import pandas as pd

# tvdatafeed v2.x
from tvDatafeed import TvDatafeed, Interval

# If you want to authenticate (helps with rate limits / coverage), add your credentials:

# Otherwise no-login mode:
tv = TvDatafeed()

# --- Category definitions ----------------------------------------------------
# All values are either a single exchange string or an iterable of exchanges to try.

# Curated Top 30 futures (no micros), TradingView continuous front-month (1!), UN-adjusted.
FUTURES_TOP30 = {
    # Equity index (CME/CBOT)
    "ES1!": "CME_MINI",   # S&P 500
    "NQ1!": "CME_MINI",   # Nasdaq 100
    "YM1!": "CBOT",       # Dow
    "RTY1!": "CME_MINI",  # Russell 2000

    # Energies (NYMEX)
    "CL1!": "NYMEX",      # WTI Crude
    "NG1!": "NYMEX",      # Natural Gas

    # Metals (COMEX)
    "GC1!": "COMEX",      # Gold
    "SI1!": "COMEX",      # Silver
    "HG1!": "COMEX",      # Copper

    # US rates (CBOT)
    "ZB1!": "CBOT",       # 30Y Bond
    "ZN1!": "CBOT",       # 10Y Note
    "ZF1!": "CBOT",       # 5Y Note
    "ZT1!": "CBOT",       # 2Y Note

    # Ags (CBOT)
    "ZC1!": "CBOT",       # Corn
    "ZS1!": "CBOT",       # Soybeans
    "ZW1!": "CBOT",       # Wheat

    # Meats (CME)
    "LE1!": "CME",        # Live Cattle
    "HE1!": "CME",        # Lean Hogs
    "GF1!": "CME",        # Feeder Cattle

    # FX futures (CME)
    "6E1!": "CME",        # Euro
    "6B1!": "CME",        # British Pound
    "6J1!": "CME",        # Japanese Yen
    "6A1!": "CME",        # Australian Dollar
    "6C1!": "CME",        # Canadian Dollar
    "6N1!": "CME",        # New Zealand Dollar

    # Dollar index (ICE US)
    "DX1!": "ICEUS",      # US Dollar Index

    # Softs (ICE US)
    "KC1!": "ICEUS",      # Coffee
    "SB1!": "ICEUS",      # Sugar
    "CC1!": "ICEUS",      # Cocoa
    "CT1!": "ICEUS",      # Cotton
}

# Top 20 cryptos by volume (Binance first, other venues as fallback)
CRYPTO_TOP20 = {
    "BTCUSDT": ["BINANCE", "BINANCEUS", "COINBASE"],
    "ETHUSDT": ["BINANCE", "BINANCEUS", "COINBASE"],
    "BNBUSDT": ["BINANCE", "BINANCEUS"],
    "SOLUSDT": ["BINANCE", "BINANCEUS"],
    "XRPUSDT": ["BINANCE", "BINANCEUS"],
    "DOGEUSDT": ["BINANCE", "BINANCEUS"],
    "ADAUSDT": ["BINANCE", "BINANCEUS"],
    "TRXUSDT": ["BINANCE"],
    "AVAXUSDT": ["BINANCE"],
    "LINKUSDT": ["BINANCE", "BINANCEUS"],
    "MATICUSDT": ["BINANCE", "BINANCEUS"],
    "LTCUSDT": ["BINANCE", "BINANCEUS"],
    "BCHUSDT": ["BINANCE", "BINANCEUS"],
    "DOTUSDT": ["BINANCE"],
    "UNIUSDT": ["BINANCE", "BINANCEUS"],
    "XLMUSDT": ["BINANCE", "BINANCEUS"],
    "ATOMUSDT": ["BINANCE"],
    "ETCUSDT": ["BINANCE", "BINANCEUS"],
    "FILUSDT": ["BINANCE"],
    "APTUSDT": ["BINANCE"],
}

# 20 common FX pairs (IDC first, OANDA as fallback)
FX_TOP20 = {
    "EURUSD": ["FX_IDC", "OANDA"],
    "USDJPY": ["FX_IDC", "OANDA"],
    "GBPUSD": ["FX_IDC", "OANDA"],
    "AUDUSD": ["FX_IDC", "OANDA"],
    "USDCAD": ["FX_IDC", "OANDA"],
    "USDCHF": ["FX_IDC", "OANDA"],
    "NZDUSD": ["FX_IDC", "OANDA"],
    "EURJPY": ["FX_IDC", "OANDA"],
    "EURGBP": ["FX_IDC", "OANDA"],
    "EURCHF": ["FX_IDC", "OANDA"],
    "EURAUD": ["FX_IDC", "OANDA"],
    "EURCAD": ["FX_IDC", "OANDA"],
    "GBPJPY": ["FX_IDC", "OANDA"],
    "GBPCHF": ["FX_IDC", "OANDA"],
    "AUDJPY": ["FX_IDC", "OANDA"],
    "CADJPY": ["FX_IDC", "OANDA"],
    "CHFJPY": ["FX_IDC", "OANDA"],
    "AUDNZD": ["FX_IDC", "OANDA"],
    "AUDCAD": ["FX_IDC", "OANDA"],
    "NZDJPY": ["FX_IDC", "OANDA"],
}

# 20 headline equity indices (cash), with generous exchange fallbacks
INDEX_TOP20 = {
    "SPX": ["CBOE", "TVC"],           # S&P 500
    "NDX": ["NASDAQ", "TVC"],       # Nasdaq 100
    "DJI": ["DJI", "TVC"],         # Dow Jones
    "RUT": ["CBOE", "TVC"],        # Russell 2000
    "VIX": ["CBOE", "TVC"],        # Volatility index
    "HSI": ["HSI", "TVC"],         # Hang Seng
    "HSCEI": ["HSI", "TVC"],       # Hang Seng China Enterprises
    "UKX": ["TVC", "LSE"],         # FTSE 100
    "DAX": ["XETR", "TVC"],        # DAX 40
    "CAC40": ["EURONEXT", "TVC"],  # CAC 40
    "EU50": ["TVC", "EUREX"],      # Euro Stoxx 50
    "IBEX": ["BME", "TVC"],        # IBEX 35
    "SMI": ["SIX", "TVC"],         # Swiss Market Index
    "AEX": ["EURONEXT", "TVC"],    # AEX 25
    "OMXS30": ["OMX", "INDEX"],    # Sweden 30
    "OMXC25": ["OMXCOP", "INDEX"], # Denmark 25
    "XJO": ["ASX", "INDEX"],       # ASX 200
    "NI225": ["TVC", "INDEX"],     # Nikkei 225
    "KOSPI": ["KRX", "INDEX"],     # KOSPI
    "NIFTY": ["NSE", "INDEX"],     # Nifty 50
}

# S&P 500 cash equities (symbol -> company name). Will try NASDAQ, NYSE, AMEX, then default.
SP500_SYMBOLS = {
    "A": "Agilent Technologies",
    "AAPL": "Apple Inc.",
    "ABBV": "AbbVie",
    "ABNB": "Airbnb",
    "ABT": "Abbott Laboratories",
    "ACGL": "Arch Capital Group",
    "ACN": "Accenture",
    "ADBE": "Adobe Inc.",
    "ADI": "Analog Devices",
    "ADM": "Archer Daniels Midland",
    "ADP": "Automatic Data Processing",
    "ADSK": "Autodesk",
    "AEE": "Ameren",
    "AEP": "American Electric Power",
    "AES": "AES Corporation",
    "AFL": "Aflac",
    "AIG": "American International Group",
    "AIZ": "Assurant",
    "AJG": "Arthur J. Gallagher & Co.",
    "AKAM": "Akamai Technologies",
    "ALB": "Albemarle Corporation",
    "ALGN": "Align Technology",
    "ALL": "Allstate",
    "ALLE": "Allegion",
    "AMAT": "Applied Materials",
    "AMCR": "Amcor",
    "AMD": "Advanced Micro Devices",
    "AME": "Ametek",
    "AMGN": "Amgen",
    "AMP": "Ameriprise Financial",
    "AMT": "American Tower",
    "AMZN": "Amazon",
    "ANET": "Arista Networks",
    "AON": "Aon plc",
    "AOS": "A. O. Smith",
    "APA": "APA Corporation",
    "APD": "Air Products",
    "APH": "Amphenol",
    "APO": "Apollo Global Management",
    "APTV": "Aptiv",
    "ARE": "Alexandria Real Estate Equities",
    "ATO": "Atmos Energy",
    "AVB": "AvalonBay Communities",
    "AVGO": "Broadcom",
    "AVY": "Avery Dennison",
    "AWK": "American Water Works",
    "AXON": "Axon Enterprise",
    "AXP": "American Express",
    "AZO": "AutoZone",
    "BA": "Boeing",
    "BAC": "Bank of America",
    "BALL": "Ball Corporation",
    "BAX": "Baxter International",
    "BBY": "Best Buy",
    "BDX": "Becton Dickinson",
    "BEN": "Franklin Resources",
    "BF.B": "Brown–Forman",
    "BG": "Bunge Global",
    "BIIB": "Biogen",
    "BK": "BNY Mellon",
    "BKNG": "Booking Holdings",
    "BKR": "Baker Hughes",
    "BLDR": "Builders FirstSource",
    "BLK": "BlackRock",
    "BMY": "Bristol Myers Squibb",
    "BR": "Broadridge Financial Solutions",
    "BRK.B": "Berkshire Hathaway",
    "BRO": "Brown & Brown",
    "BSX": "Boston Scientific",
    "BX": "Blackstone Inc.",
    "BXP": "BXP, Inc.",
    "C": "Citigroup",
    "CAG": "Conagra Brands",
    "CAH": "Cardinal Health",
    "CARR": "Carrier Global",
    "CAT": "Caterpillar Inc.",
    "CB": "Chubb Limited",
    "CBOE": "Cboe Global Markets",
    "CBRE": "CBRE Group",
    "CCI": "Crown Castle",
    "CCL": "Carnival",
    "CDNS": "Cadence Design Systems",
    "CDW": "CDW Corporation",
    "CEG": "Constellation Energy",
    "CF": "CF Industries",
    "CFG": "Citizens Financial Group",
    "CHD": "Church & Dwight",
    "CHRW": "C.H. Robinson",
    "CHTR": "Charter Communications",
    "CI": "Cigna",
    "CINF": "Cincinnati Financial",
    "CL": "Colgate-Palmolive",
    "CLX": "Clorox",
    "CMCSA": "Comcast",
    "CME": "CME Group",
    "CMG": "Chipotle Mexican Grill",
    "CMI": "Cummins",
    "CMS": "CMS Energy",
    "CNC": "Centene Corporation",
    "CNP": "CenterPoint Energy",
    "COF": "Capital One",
    "COIN": "Coinbase",
    "COO": "Cooper Companies (The)",
    "COP": "ConocoPhillips",
    "COR": "Cencora",
    "COST": "Costco",
    "CPAY": "Corpay",
    "CPB": "Campbell's Company (The)",
    "CPRT": "Copart",
    "CPT": "Camden Property Trust",
    "CRL": "Charles River Laboratories",
    "CRM": "Salesforce",
    "CRWD": "CrowdStrike",
    "CSCO": "Cisco",
    "CSGP": "CoStar Group",
    "CSX": "CSX Corporation",
    "CTAS": "Cintas",
    "CTRA": "Coterra",
    "CTSH": "Cognizant",
    "CTVA": "Corteva",
    "CVS": "CVS Health",
    "CVX": "Chevron Corporation",
    "CZR": "Caesars Entertainment",
    "D": "Dominion Energy",
    "DAL": "Delta Air Lines",
    "DASH": "DoorDash",
    "DAY": "Dayforce",
    "DD": "DuPont",
    "DDOG": "Datadog",
    "DE": "Deere & Company",
    "DECK": "Deckers Brands",
    "DELL": "Dell Technologies",
    "DG": "Dollar General",
    "DGX": "Quest Diagnostics",
    "DHI": "D. R. Horton",
    "DHR": "Danaher Corporation",
    "DIS": "Walt Disney Company (The)",
    "DLR": "Digital Realty",
    "DLTR": "Dollar Tree",
    "DOC": "Healthpeak Properties",
    "DOV": "Dover Corporation",
    "DOW": "Dow Inc.",
    "DPZ": "Domino's",
    "DRI": "Darden Restaurants",
    "DTE": "DTE Energy",
    "DUK": "Duke Energy",
    "DVA": "DaVita",
    "DVN": "Devon Energy",
    "DXCM": "Dexcom",
    "EA": "Electronic Arts",
    "EBAY": "eBay Inc.",
    "ECL": "Ecolab",
    "ED": "Consolidated Edison",
    "EFX": "Equifax",
    "EG": "Everest Group",
    "EIX": "Edison International",
    "EL": "Estée Lauder Companies (The)",
    "ELV": "Elevance Health",
    "EMN": "Eastman Chemical Company",
    "EMR": "Emerson Electric",
    "ENPH": "Enphase Energy",
    "EOG": "EOG Resources",
    "EPAM": "EPAM Systems",
    "EQIX": "Equinix",
    "EQR": "Equity Residential",
    "EQT": "EQT Corporation",
    "ERIE": "Erie Indemnity",
    "ES": "Eversource Energy",
    "ESS": "Essex Property Trust",
    "ETN": "Eaton Corporation",
    "ETR": "Entergy",
    "EVRG": "Evergy",
    "EW": "Edwards Lifesciences",
    "EXC": "Exelon",
    "EXE": "Expand Energy",
    "EXPD": "Expeditors International",
    "EXPE": "Expedia Group",
    "EXR": "Extra Space Storage",
    "F": "Ford Motor Company",
    "FANG": "Diamondback Energy",
    "FAST": "Fastenal",
    "FCX": "Freeport-McMoRan",
    "FDS": "FactSet",
    "FDX": "FedEx",
    "FE": "FirstEnergy",
    "FFIV": "F5, Inc.",
    "FI": "Fiserv",
    "FICO": "Fair Isaac",
    "FIS": "Fidelity National Information Services",
    "FITB": "Fifth Third Bancorp",
    "FOX": "Fox Corporation (Class B)",
    "FOXA": "Fox Corporation (Class A)",
    "FRT": "Federal Realty Investment Trust",
    "FSLR": "First Solar",
    "FTNT": "Fortinet",
    "FTV": "Fortive",
    "GD": "General Dynamics",
    "GDDY": "GoDaddy",
    "GE": "GE Aerospace",
    "GEHC": "GE HealthCare",
    "GEN": "Gen Digital",
    "GEV": "GE Vernova",
    "GILD": "Gilead Sciences",
    "GIS": "General Mills",
    "GL": "Globe Life",
    "GLW": "Corning Inc.",
    "GM": "General Motors",
    "GNRC": "Generac",
    "GOOG": "Alphabet Inc. (Class C)",
    "GOOGL": "Alphabet Inc. (Class A)",
    "GPC": "Genuine Parts Company",
    "GPN": "Global Payments",
    "GRMN": "Garmin",
    "GS": "Goldman Sachs",
    "GWW": "W. W. Grainger",
    "HAL": "Halliburton",
    "HAS": "Hasbro",
    "HBAN": "Huntington Bancshares",
    "HCA": "HCA Healthcare",
    "HD": "Home Depot (The)",
    "HIG": "Hartford (The)",
    "HII": "Huntington Ingalls Industries",
    "HLT": "Hilton Worldwide",
    "HOLX": "Hologic",
    "HON": "Honeywell",
    "HPE": "Hewlett Packard Enterprise",
    "HPQ": "HP Inc.",
    "HRL": "Hormel Foods",
    "HSIC": "Henry Schein",
    "HST": "Host Hotels & Resorts",
    "HSY": "Hershey Company (The)",
    "HUBB": "Hubbell Incorporated",
    "HUM": "Humana",
    "HWM": "Howmet Aerospace",
    "IBM": "IBM",
    "ICE": "Intercontinental Exchange",
    "IDXX": "Idexx Laboratories",
    "IEX": "IDEX Corporation",
    "IFF": "International Flavors & Fragrances",
    "INCY": "Incyte",
    "INTC": "Intel",
    "INTU": "Intuit",
    "INVH": "Invitation Homes",
    "IP": "International Paper",
    "IPG": "Interpublic Group of Companies (The)",
    "IQV": "IQVIA",
    "IR": "Ingersoll Rand",
    "IRM": "Iron Mountain",
    "ISRG": "Intuitive Surgical",
    "IT": "Gartner",
    "ITW": "Illinois Tool Works",
    "IVZ": "Invesco",
    "J": "Jacobs Solutions",
    "JBHT": "J.B. Hunt",
    "JBL": "Jabil",
    "JCI": "Johnson Controls",
    "JKHY": "Jack Henry & Associates",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "K": "Kellanova",
    "KDP": "Keurig Dr Pepper",
    "KEY": "KeyCorp",
    "KEYS": "Keysight Technologies",
    "KHC": "Kraft Heinz",
    "KIM": "Kimco Realty",
    "KKR": "KKR & Co.",
    "KLAC": "KLA Corporation",
    "KMB": "Kimberly-Clark",
    "KMI": "Kinder Morgan",
    "KMX": "CarMax",
    "KO": "Coca-Cola Company (The)",
    "KR": "Kroger",
    "KVUE": "Kenvue",
    "L": "Loews Corporation",
    "LDOS": "Leidos",
    "LEN": "Lennar",
    "LH": "Labcorp",
    "LHX": "L3Harris",
    "LII": "Lennox International",
    "LIN": "Linde plc",
    "LKQ": "LKQ Corporation",
    "LLY": "Lilly (Eli)",
    "LMT": "Lockheed Martin",
    "LNT": "Alliant Energy",
    "LOW": "Lowe's",
    "LRCX": "Lam Research",
    "LULU": "Lululemon Athletica",
    "LUV": "Southwest Airlines",
    "LVS": "Las Vegas Sands",
    "LW": "Lamb Weston",
    "LYB": "LyondellBasell",
    "LYV": "Live Nation Entertainment",
    "MA": "Mastercard",
    "MAA": "Mid-America Apartment Communities",
    "MAR": "Marriott International",
    "MAS": "Masco",
    "MCD": "McDonald's",
    "MCHP": "Microchip Technology",
    "MCK": "McKesson Corporation",
    "MCO": "Moody's Corporation",
    "MDLZ": "Mondelez International",
    "MDT": "Medtronic",
    "MET": "MetLife",
    "META": "Meta Platforms",
    "MGM": "MGM Resorts",
    "MHK": "Mohawk Industries",
    "MKC": "McCormick & Company",
    "MKTX": "MarketAxess",
    "MLM": "Martin Marietta Materials",
    "MMC": "Marsh McLennan",
    "MMM": "3M",
    "MNST": "Monster Beverage",
    "MO": "Altria Group",
    "MOH": "Molina Healthcare",
    "MOS": "Mosaic Company (The)",
    "MPC": "Marathon Petroleum",
    "MPWR": "Monolithic Power Systems",
    "MRK": "Merck & Co.",
    "MRNA": "Moderna",
    "MRO": "Marathon Oil",
    "MS": "Morgan Stanley",
    "MSCI": "MSCI Inc.",
    "MSFT": "Microsoft",
    "MSI": "Motorola Solutions",
    "MTB": "M&T Bank",
    "MTCH": "Match Group",
    "MTD": "Mettler Toledo",
    "MU": "Micron Technology",
    "NCLH": "Norwegian Cruise Line",
    "NDAQ": "Nasdaq, Inc.",
    "NDSN": "Nordson Corporation",
    "NEE": "NextEra Energy",
    "NEM": "Newmont",
    "NFLX": "Netflix",
    "NI": "NiSource",
    "NKE": "Nike",
    "NOC": "Northrop Grumman",
    "NOW": "ServiceNow",
    "NRG": "NRG Energy",
    "NSC": "Norfolk Southern",
    "NTAP": "NetApp",
    "NTRS": "Northern Trust",
    "NUE": "Nucor",
    "NVDA": "Nvidia",
    "NVR": "NVR, Inc.",
    "NWS": "News Corp (Class B)",
    "NWSA": "News Corp (Class A)",
    "NXPI": "NXP Semiconductors",
    "O": "Realty Income",
    "ODFL": "Old Dominion",
    "OKE": "ONEOK",
    "OMC": "Omnicom Group",
    "ON": "ON Semiconductor",
    "ORCL": "Oracle Corporation",
    "ORLY": "O'Reilly Automotive",
    "OTIS": "Otis Worldwide",
    "OXY": "Occidental Petroleum",
    "PANW": "Palo Alto Networks",
    "PARA": "Paramount Global",
    "PCG": "PG&E Corporation",
    "PEG": "Public Service Enterprise Group",
    "PEP": "PepsiCo",
    "PFE": "Pfizer",
    "PFG": "Principal Financial Group",
    "PG": "Procter & Gamble",
    "PGR": "Progressive Corporation",
    "PH": "Parker Hannifin",
    "PHM": "PulteGroup",
    "PKG": "Packaging Corporation of America",
    "PLD": "Prologis",
    "PLTR": "Palantir Technologies",
    "PM": "Philip Morris International",
    "PNC": "PNC Financial Services",
    "PNR": "Pentair",
    "PNW": "Pinnacle West Capital",
    "PODD": "Insulet Corporation",
    "POOL": "Pool Corporation",
    "PPG": "PPG Industries",
    "PPL": "PPL Corporation",
    "PRU": "Prudential Financial",
    "PSA": "Public Storage",
    "PSX": "Phillips 66",
    "PTC": "PTC Inc.",
    "PWR": "Quanta Services",
    "PYPL": "PayPal",
    "QCOM": "Qualcomm",
    "QRVO": "Qorvo",
    "RCL": "Royal Caribbean Group",
    "REG": "Regency Centers",
    "REGN": "Regeneron Pharmaceuticals",
    "RF": "Regions Financial",
    "RJF": "Raymond James",
    "RL": "Ralph Lauren Corporation",
    "RMD": "ResMed",
    "ROK": "Rockwell Automation",
    "ROL": "Rollins, Inc.",
    "ROP": "Roper Technologies",
    "ROST": "Ross Stores",
    "RSG": "Republic Services",
    "RTX": "RTX Corporation",
    "RVTY": "Revvity",
    "SBAC": "SBA Communications",
    "SBUX": "Starbucks",
    "SCHW": "Charles Schwab Corporation",
    "SHW": "Sherwin-Williams",
    "SJM": "J.M. Smucker Company (The)",
    "SLB": "Schlumberger",
    "SMCI": "Super Micro Computer",
    "SNA": "Snap-on",
    "SNPS": "Synopsys",
    "SO": "Southern Company",
    "SOLV": "Solventum",
    "SPG": "Simon Property Group",
    "SPGI": "S&P Global",
    "SRE": "Sempra",
    "STE": "Steris",
    "STLD": "Steel Dynamics",
    "STT": "State Street Corporation",
    "STX": "Seagate Technology",
    "STZ": "Constellation Brands",
    "SWK": "Stanley Black & Decker",
    "SWKS": "Skyworks Solutions",
    "SYF": "Synchrony Financial",
    "SYK": "Stryker Corporation",
    "SYY": "Sysco",
    "T": "AT&T",
    "TAP": "Molson Coors Beverage Company",
    "TDG": "TransDigm Group",
    "TDY": "Teledyne Technologies",
    "TECH": "Bio-Techne",
    "TEL": "TE Connectivity",
    "TER": "Teradyne",
    "TFC": "Truist Financial",
    "TFX": "Teleflex",
    "TGT": "Target Corporation",
    "TJX": "TJX Companies",
    "TMO": "Thermo Fisher Scientific",
    "TMUS": "T-Mobile US",
    "TPR": "Tapestry, Inc.",
    "TRGP": "Targa Resources",
    "TRMB": "Trimble Inc.",
    "TROW": "T. Rowe Price",
    "TRV": "Travelers Companies (The)",
    "TSCO": "Tractor Supply Company",
    "TSLA": "Tesla, Inc.",
    "TSN": "Tyson Foods",
    "TT": "Trane Technologies",
    "TTWO": "Take-Two Interactive",
    "TXN": "Texas Instruments",
    "TXT": "Textron",
    "TYL": "Tyler Technologies",
    "UAL": "United Airlines",
    "UBER": "Uber Technologies",
    "UDR": "UDR, Inc.",
    "UHS": "Universal Health Services",
    "ULTA": "Ulta Beauty",
    "UNH": "UnitedHealth Group",
    "UNP": "Union Pacific",
    "UPS": "United Parcel Service",
    "URI": "United Rentals",
    "USB": "U.S. Bancorp",
    "V": "Visa Inc.",
    "VICI": "VICI Properties",
    "VLO": "Valero Energy",
    "VLTO": "Veralto Corporation",
    "VMC": "Vulcan Materials",
    "VRSK": "Verisk Analytics",
    "VRSN": "VeriSign",
    "VRTX": "Vertex Pharmaceuticals",
    "VST": "Vistra Corp",
    "VTR": "Ventas",
    "VTRS": "Viatris",
    "VZ": "Verizon",
    "WAB": "Westinghouse Air Brake Technologies",
    "WAT": "Waters Corporation",
    "WBA": "Walgreens Boots Alliance",
    "WBD": "Warner Bros. Discovery",
    "WDC": "Western Digital",
    "WEC": "WEC Energy Group",
    "WELL": "Welltower",
    "WFC": "Wells Fargo",
    "WM": "Waste Management",
    "WMB": "Williams Companies",
    "WMT": "Walmart",
    "WRB": "W. R. Berkley Corporation",
    "WRK": "WestRock",
    "WST": "West Pharmaceutical Services",
    "WTW": "Willis Towers Watson",
    "WY": "Weyerhaeuser",
    "WYNN": "Wynn Resorts",
    "XEL": "Xcel Energy",
    "XOM": "Exxon Mobil",
    "XYL": "Xylem Inc.",
    "YUM": "Yum! Brands",
    "ZBH": "Zimmer Biomet",
    "ZBRA": "Zebra Technologies",
    "ZTS": "Zoetis",
}
SP500_EXCHANGES = ("NASDAQ", "NYSE", "AMEX", "")
SP500_SYMBOLS_WITH_EXCHANGES = {sym: SP500_EXCHANGES for sym in SP500_SYMBOLS}

OUTDIR_ROOT = Path("data_tv_downloads")
OUTDIR_ROOT.mkdir(parents=True, exist_ok=True)


def _normalize_exchanges(exchanges: str | Iterable[str]) -> list[str]:
    if isinstance(exchanges, str):
        return [exchanges]
    try:
        seq = list(exchanges)
        return seq if seq else [""]
    except TypeError:
        return [""]


def fetch_symbol(tv: TvDatafeed, sym: str, exchanges: str | Iterable[str], n_bars: int = 5000, company: str | None = None) -> tuple[pd.DataFrame, str]:
    """
    Fetch daily bars for a TradingView symbol, trying multiple exchanges until one works.
    Returns (DataFrame, exchange_used).
    """
    for exch in _normalize_exchanges(exchanges):
        try:
            df = tv.get_hist(
                symbol=sym,
                exchange=exch,
                interval=Interval.in_daily,
                n_bars=n_bars,
            )
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns={
                    "symbol": "tv_symbol",
                    "datetime": "date",
                })
                df.insert(0, "symbol", sym)
                df.insert(1, "exchange", exch)
                if company is not None:
                    df.insert(2, "company", company)
                df = df.sort_values("date").reset_index(drop=True)
                return df, exch
        except Exception:
            continue
    return pd.DataFrame(), ""


def process_category(name: str, symbols: dict[str, str | Iterable[str]], n_bars: int = 5000, cooldown: float = 0.4, symbol_to_company: dict[str, str] | None = None) -> tuple[list[pd.DataFrame], list[tuple[str, str, str]]]:
    outdir = OUTDIR_ROOT / name
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    errors: list[tuple[str, str, str]] = []

    for i, (sym, exchanges) in enumerate(symbols.items(), start=1):
        try:
            exch_list = _normalize_exchanges(exchanges)
            exch_label = exch_list[0] if exch_list else ""
            print(f"[{name} {i:02d}/{len(symbols)}] Fetching {exch_label}:{sym} …")
            company = symbol_to_company.get(sym) if symbol_to_company else None
            df, used_exch = fetch_symbol(tv, sym, exchanges, n_bars=n_bars, company=company)
            if df.empty:
                errors.append((sym, ",".join(exch_list), "empty"))
                print(f"   -> No data returned for {sym}")
                time.sleep(cooldown)
                continue

            sanitized = sym.replace("!", "bang")
            out = outdir / f"{sanitized}_{used_exch or 'default'}_D_{n_bars}.csv"
            df.to_csv(out, index=False)
            print(f"   -> Wrote {out.name} ({len(df)} rows) via {used_exch or 'default'}")

            all_rows.append(df)
            time.sleep(cooldown)

        except Exception as e:
            errors.append((sym, ",".join(_normalize_exchanges(exchanges)), str(e)))
            print(f"   -> ERROR {sym}: {e}")
            time.sleep(max(cooldown, 1.0))

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_out = outdir / f"{name}_combined.csv"
        combined.to_csv(combined_out, index=False)
        print(f"   -> Combined CSV written: {combined_out} (rows={len(combined)})")

    return all_rows, errors


def main() -> None:
    # (category_name, symbol_map, n_bars, cooldown, optional_company_lookup)
    categories = [
        ("futures_top30", FUTURES_TOP30, 5000, 0.4, None),
        ("crypto_top20", CRYPTO_TOP20, 5000, 0.4, None),
        ("fx_top20", FX_TOP20, 5000, 0.4, None),
        ("indices_top20", INDEX_TOP20, 5000, 0.5, None),
        ("sp500", SP500_SYMBOLS_WITH_EXCHANGES, 5000, 0.5, SP500_SYMBOLS),
    ]

    grand_errors: list[tuple[str, str, str, str]] = []

    for name, mapping, n_bars, cooldown, company_lookup in categories:
        print(f"\n=== Downloading {name} ===")
        _, errors = process_category(
            name,
            mapping,
            n_bars=n_bars,
            cooldown=cooldown,
            symbol_to_company=company_lookup,
        )
        for sym, exch, msg in errors:
            grand_errors.append((name, sym, exch, msg))

    if grand_errors:
        print("\nCompleted with some issues:")
        for cat, sym, exch, msg in grand_errors:
            print(f" - [{cat}] {sym} ({exch}) -> {msg}")
    else:
        print("\nAll symbols fetched successfully across all categories.")


if __name__ == "__main__":
    main()
