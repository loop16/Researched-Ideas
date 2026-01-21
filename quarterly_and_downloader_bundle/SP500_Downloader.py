from pathlib import Path
import time
import pandas as pd
from datetime import datetime
import logging
import json

# tvdatafeed v2.x
from tvDatafeed import TvDatafeed, Interval

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sp500_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize TvDatafeed
# Uncomment and add credentials if you have them (helps with rate limits)
# tv = TvDatafeed(username='your_username', password='your_password')
tv = TvDatafeed()

# S&P 500 stocks with their company names
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
    "ZTS": "Zoetis"
}

# Output directory
OUTDIR = Path("data_sp500_daily")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Exchange cache file
EXCHANGE_CACHE_FILE = OUTDIR / "exchange_cache.json"

# Load exchange cache if it exists
def load_exchange_cache():
    """Load the exchange cache from file"""
    if EXCHANGE_CACHE_FILE.exists():
        try:
            with open(EXCHANGE_CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

# Save exchange cache
def save_exchange_cache(cache):
    """Save the exchange cache to file"""
    with open(EXCHANGE_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

# Global exchange cache
EXCHANGE_CACHE = load_exchange_cache()

# Rate limiting configuration
class RateLimiter:
    def __init__(self, requests_per_minute=30, burst_delay=0.5):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute: Maximum requests per minute
            burst_delay: Delay between individual requests in seconds
        """
        self.requests_per_minute = requests_per_minute
        self.burst_delay = burst_delay
        self.request_times = []
        
    def wait(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # If we've hit the per-minute limit, wait until we can make another request
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0]) + 1
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.request_times = []
        
        # Always wait the burst delay between requests
        time.sleep(self.burst_delay)
        
        # Record this request
        self.request_times.append(time.time())

def fetch_symbol(tv: TvDatafeed, sym: str, n_bars: int = 5000) -> tuple[pd.DataFrame, str]:
    """
    Fetch daily bars for a stock symbol from TradingView
    
    Args:
        tv: TvDatafeed instance
        sym: Stock symbol
        n_bars: Number of bars to fetch
    
    Returns:
        Tuple of (DataFrame with OHLCV data, exchange name that worked)
    """
    df = None
    successful_exchange = None
    
    # Check cache first
    if sym in EXCHANGE_CACHE:
        exchanges_to_try = [EXCHANGE_CACHE[sym]] + [e for e in ['NASDAQ', 'NYSE', 'AMEX', ''] if e != EXCHANGE_CACHE[sym]]
        logger.debug(f"   -> Using cached exchange: {EXCHANGE_CACHE[sym]}")
    else:
        exchanges_to_try = ['NASDAQ', 'NYSE', 'AMEX', '']
    
    for exchange in exchanges_to_try:
        try:
            df = tv.get_hist(
                symbol=sym,
                exchange=exchange,
                interval=Interval.in_daily,
                n_bars=n_bars
            )
            if df is not None and not df.empty:
                successful_exchange = exchange
                exchange_label = exchange if exchange else 'default'
                logger.debug(f"   -> Found data on {exchange_label} exchange")
                
                # Update cache
                if sym not in EXCHANGE_CACHE or EXCHANGE_CACHE[sym] != exchange:
                    EXCHANGE_CACHE[sym] = exchange
                    save_exchange_cache(EXCHANGE_CACHE)
                break
        except Exception as e:
            logger.debug(f"   -> {exchange if exchange else 'default'} exchange failed: {str(e)[:50]}")
            continue
    
    if df is None or df.empty:
        return pd.DataFrame(), None
    
    # Standardize columns & attach metadata
    df = df.reset_index().rename(columns={
        "symbol": "tv_symbol",
        "datetime": "date"
    })
    df.insert(0, "symbol", sym)
    df.insert(1, "exchange", successful_exchange if successful_exchange else "")
    df.insert(2, "company", SP500_SYMBOLS.get(sym, "Unknown"))
    
    # Ensure sorted by date ascending
    df = df.sort_values("date").reset_index(drop=True)
    return df, successful_exchange

def main():
    """Main download function with rate limiting and error handling"""
    
    # Initialize rate limiter (60 requests per minute, 1 sec between requests)
    rate_limiter = RateLimiter(requests_per_minute=60, burst_delay=1.0)
    
    all_rows = []
    errors = []
    successful = 0
    
    total = len(SP500_SYMBOLS)
    start_time = time.time()
    
    logger.info(f"Starting download of {total} S&P 500 stocks...")
    logger.info(f"Output directory: {OUTDIR.absolute()}")
    
    for i, (sym, company) in enumerate(SP500_SYMBOLS.items(), start=1):
        try:
            # Apply rate limiting
            rate_limiter.wait()
            
            logger.info(f"[{i:03d}/{total}] Fetching {sym} ({company})...")
            df, exchange = fetch_symbol(tv, sym, n_bars=5000)
            
            if df.empty:
                logger.warning(f"   -> No data returned for {sym}")
                errors.append((sym, company, "empty"))
                continue
            
            # Save per-symbol CSV
            out = OUTDIR / f"{sym}_daily.csv"
            df.to_csv(out, index=False)
            exchange_label = exchange if exchange else 'default'
            logger.info(f"   -> Wrote {out.name} ({len(df)} rows) from {exchange_label}")
            
            all_rows.append(df)
            successful += 1
            
            # Progress update every 50 symbols
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed * 60
                remaining = (total - i) / rate if rate > 0 else 0
                logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%) - Rate: {rate:.1f} symbols/min - ETA: {remaining:.1f} min")
        
        except Exception as e:
            logger.error(f"   -> ERROR {sym} -> {e}")
            errors.append((sym, company, str(e)))
            # Extra backoff on error
            time.sleep(2.0)
    
    # Write combined CSV if anything succeeded
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_out = OUTDIR / "sp500_combined_daily.csv"
        combined.to_csv(combined_out, index=False)
        logger.info(f"\nCombined CSV written: {combined_out} (rows={len(combined)})")
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info(f"Download completed in {elapsed_time/60:.1f} minutes")
    logger.info(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    logger.info(f"Failed: {len(errors)}/{total}")
    
    if errors:
        logger.info("\nFailed symbols:")
        for sym, company, msg in errors[:20]:  # Show first 20 errors
            logger.info(f" - {sym} ({company}) -> {msg}")
        if len(errors) > 20:
            logger.info(f" ... and {len(errors)-20} more errors")
        
        # Save errors to file
        error_df = pd.DataFrame(errors, columns=['Symbol', 'Company', 'Error'])
        error_file = OUTDIR / "download_errors.csv"
        error_df.to_csv(error_file, index=False)
        logger.info(f"\nFull error list saved to: {error_file}")
    else:
        logger.info("\nAll symbols fetched successfully! 🎉")
    
    logger.info("="*80)

if __name__ == "__main__":
    main()
