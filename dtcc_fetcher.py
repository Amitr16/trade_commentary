#!/usr/bin/env python3
"""
Independent DTCC Data Fetcher
Fetches trade data from DTCC API, processes it, and saves to CSV with duplicate handling.
Handles trade modifications by replacing old trades when Original Dissemination Identifier is present.
"""

import requests
import csv
import os
import time
import logging
import math
from datetime import datetime
from typing import Dict, List, Set, Optional
import threading
import signal
import sys
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dtcc_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DTCC_API_URLS = {
    'CFTC': "https://pddata.dtcc.com/ppd/api/ticker/CFTC/RATES",
    'SEC': "https://pddata.dtcc.com/ppd/api/ticker/SEC/RATES", 
    'CANADA': "https://pddata.dtcc.com/ppd/api/ticker/canada/RATES"
}
CSV_FILE_NAME = "dtcc_trades.csv"
FETCH_INTERVAL = 60  # seconds
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

class DTCCFetcher:
    def __init__(self, csv_file: str = CSV_FILE_NAME):
        self.csv_file = csv_file
        self.running = False
        self.thread = None
        self.existing_dissemination_ids: Set[str] = set()
        self.existing_original_dissemination_ids: Set[str] = set()
        
        # CSV field names
        self.fieldnames = [
            'Source', 'Trade Time', 'Effective Date', 'Expiration Date', 'Effective Bucket', 'Expiration Bucket', 'Tenor', 'Currency',
            'Rates', 'Notionals', 'DV01', 'Frequency', 'Action Type', 'Event Type',
            'Asset Class', 'UPI Underlier Name', 'Unique Product Identifier',
            'Dissemination Identifier', 'Original Dissemination Identifier', 
            'Other Payment Type', 'Package Indicator',
            'Floating Rate Payment Frequency Period Leg2', 
            'Floating Rate Payment Frequency Period Multiplier Leg2',
            'Fixed Rate Payment Frequency Period Leg1', 
            'Fixed Rate Payment Frequency Period Multiplier Leg1',
            'Created At', 'Updated At'
        ]
        
        # Initialize CSV file if it doesn't exist
        self._initialize_csv()
        self._load_existing_ids()
        
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()
                logger.info(f"Created new CSV file: {self.csv_file}")
            except Exception as e:
                logger.error(f"Error creating CSV file: {e}")
                raise
    
    def _load_existing_ids(self):
        """Load existing dissemination IDs from CSV to track duplicates"""
        self.existing_dissemination_ids.clear()
        self.existing_original_dissemination_ids.clear()
        
        if not os.path.exists(self.csv_file):
            return
            
        try:
            with open(self.csv_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Track dissemination identifiers
                    if row.get('Dissemination Identifier'):
                        self.existing_dissemination_ids.add(row['Dissemination Identifier'])
                    
                    # Track original dissemination identifiers
                    if row.get('Original Dissemination Identifier'):
                        self.existing_original_dissemination_ids.add(row['Original Dissemination Identifier'])
            
            logger.info(f"Loaded {len(self.existing_dissemination_ids)} existing dissemination IDs")
            logger.info(f"Loaded {len(self.existing_original_dissemination_ids)} existing original dissemination IDs")
            
        except Exception as e:
            logger.error(f"Error loading existing IDs: {e}")
    
    def fetch_trade_data(self, source: str) -> Optional[Dict]:
        """Fetch trade data from specific DTCC API source"""
        url = DTCC_API_URLS.get(source)
        if not url:
            logger.error(f"Unknown source: {source}")
            return None
            
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Fetching data from {source} API (attempt {attempt + 1}/{MAX_RETRIES})")
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                
                data = response.json()
                trade_count = len(data.get('tradeList', []))
                logger.info(f"Successfully fetched data from {source}: {trade_count} trades")
                return data
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {source} (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5)  # Wait before retry
                else:
                    logger.error(f"All {MAX_RETRIES} attempts failed for {source}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error fetching data from {source}: {e}")
                return None
    
    def fetch_all_sources(self) -> Dict[str, List[Dict]]:
        """Fetch trade data from all DTCC API sources"""
        all_trades = {}
        
        for source in DTCC_API_URLS.keys():
            try:
                data = self.fetch_trade_data(source)
                if data and 'tradeList' in data:
                    all_trades[source] = data['tradeList']
                else:
                    logger.warning(f"No trade data received from {source}")
                    all_trades[source] = []
            except Exception as e:
                logger.error(f"Error fetching from {source}: {e}")
                all_trades[source] = []
        
        return all_trades
    
    def calculate_dv01(self, notional: float, rate: float, effective_date: datetime, expiration_date: datetime, frequency: str = 'Semi-Annual') -> float:
        """
        Calculate DV01 (Dollar Value of 01 basis point) for an interest rate swap
        DV01 = Notional × 1bp × PVBP, where PVBP = sum_i (alpha_i × DF_i)
        For flat curve and equal accruals: alpha_i = 1/m and DF_i = (1 + r/m)^(-i)
        """
        try:
            if notional <= 0 or effective_date >= expiration_date:
                return 0.0
            
            # Convert rate to decimal if it's in percentage
            if rate > 1:
                rate = rate / 100.0
            
            # Payment frequency mapping
            freq_map = {'Annual': 1, 'Semi-Annual': 2, 'Quarterly': 4, 'Monthly': 12, 'Weekly': 52, 'Daily': 365}
            m = freq_map.get(frequency, 2)
            
            # Total years
            T = (expiration_date - effective_date).days / 365.25
            n = int(round(m * T))
            if n <= 0:
                return 0.0
            
            # PVBP (Present Value of Basis Point) calculation
            pvbp = (1.0 / m) * sum((1.0 + rate / m) ** (-k) for k in range(1, n + 1))
            
            # DV01 calculation
            dv01 = notional * 1e-4 * pvbp
            return round(dv01, 2)
            
        except Exception as e:
            logger.warning(f"Error calculating DV01: {e}")
            return 0.0
    
    def load_fx_rates(self):
        """Load FX conversion rates from fx.csv file"""
        fx_rates = {}
        fx_file = 'fx.csv'
        if not os.path.exists(fx_file):
            logger.warning(f"fx.csv not found, using 1.0 for all currencies")
            return {}
        
        try:
            with open(fx_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    currency = row['Currency'].strip().upper()
                    value = float(row['Value'])
                    fx_rates[currency] = value
            
            logger.info(f"Loaded FX rates for {len(fx_rates)} currencies")
            return fx_rates
            
        except Exception as e:
            logger.error(f"Error loading FX rates: {e}")
            return {}
    
    def convert_dv01_to_usd(self, dv01_local: float, currency: str, fx_rates: dict) -> float:
        """Convert DV01 from local currency to USD using FX rates"""
        if not currency or not fx_rates:
            return dv01_local
        
        currency_upper = currency.upper()
        if currency_upper == 'USD':
            return dv01_local
        
        fx_rate = fx_rates.get(currency_upper)
        if fx_rate is None:
            logger.warning(f"No FX rate found for currency {currency}, using local value")
            return dv01_local
        
        dv01_usd = dv01_local / fx_rate
        return dv01_usd
    
    def load_mpc_dates(self):
        """Load MPC dates from CSV file with robust validation"""
        mpc_dates = {}
        mpc_file = 'MPC_Dates.csv'
        
        if not os.path.exists(mpc_file):
            logger.warning(f"MPC_Dates.csv not found, skipping MPC date detection")
            return mpc_dates
        
        try:
            with open(mpc_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Validate required columns
                required_columns = ['Bank', 'Name', 'Date', 'Currency']
                if not all(col in reader.fieldnames for col in required_columns):
                    logger.error(f"MPC_Dates.csv missing required columns. Found: {reader.fieldnames}, Required: {required_columns}")
                    return {}
                
                row_count = 0
                for row in reader:
                    row_count += 1
                    
                    # Skip empty rows
                    if not any(row.values()):
                        continue
                    
                    # Extract and validate fields
                    currency = row.get('Currency', '').strip().upper()
                    date_str = row.get('Date', '').strip()
                    name = row.get('Name', '').strip()
                    bank = row.get('Bank', '').strip()
                    
                    # Validate required fields
                    if not currency or not date_str or not name:
                        logger.warning(f"Row {row_count}: Missing required fields (Currency, Date, or Name)")
                        continue
                    
                    # Parse date with multiple format support
                    date_obj = None
                    for date_format in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y']:
                        try:
                            date_obj = datetime.strptime(date_str, date_format).date()
                            break
                        except ValueError:
                            continue
                    
                    if date_obj is None:
                        logger.warning(f"Row {row_count}: Could not parse date '{date_str}'. Supported formats: MM/DD/YYYY, YYYY-MM-DD, DD/MM/YYYY")
                        continue
                    
                    # Initialize currency dictionary
                    if currency not in mpc_dates:
                        mpc_dates[currency] = {}
                    
                    # Check for duplicate dates within same currency
                    if date_obj in mpc_dates[currency]:
                        logger.warning(f"Row {row_count}: Duplicate date {date_obj} for currency {currency}. Keeping first occurrence: {mpc_dates[currency][date_obj]}")
                        continue
                    
                    # Store the mapping
                    mpc_dates[currency][date_obj] = name
                    logger.debug(f"Loaded MPC date: {currency} {date_obj} -> {name} ({bank})")
                
                logger.info(f"Loaded {sum(len(dates) for dates in mpc_dates.values())} MPC dates for currencies: {list(mpc_dates.keys())}")
                return mpc_dates
                
        except Exception as e:
            logger.error(f"Error loading MPC dates: {e}")
            return {}
    
    def load_imm_dates(self):
        """Load IMM dates from CSV file"""
        imm_dates = {}
        imm_file = 'IMM_Dates.csv'
        
        if not os.path.exists(imm_file):
            logger.warning(f"IMM_Dates.csv not found, using hardcoded IMM dates")
            # Fallback to hardcoded values
            return {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}
        
        try:
            with open(imm_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    month = int(row['Month'])
                    code = row['Code']
                    imm_dates[month] = code
            
            logger.info(f"Loaded IMM dates: {imm_dates}")
            return imm_dates
            
        except Exception as e:
            logger.error(f"Error loading IMM dates: {e}")
            # Fallback to hardcoded values
            return {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}
    
    def get_mpc_name(self, date, currency, mpc_dates):
        """Get MPC name for date and currency with robust matching"""
        if not mpc_dates or not currency or not date:
            return None
        
        # Normalize currency to uppercase
        currency_upper = str(currency).strip().upper()
        
        # Check if currency exists in MPC dates
        if currency_upper not in mpc_dates:
            return None
        
        # Check if exact date match exists
        if date in mpc_dates[currency_upper]:
            return mpc_dates[currency_upper][date]
        
        return None
    
    def get_imm_code(self, date, imm_dates):
        """Get IMM code for date (H, M, U, Z + year)"""
        if date.month in imm_dates and 15 <= date.day <= 21 and date.weekday() == 2:
            # Use two-digit year for dates beyond 9 years to avoid confusion
            current_year = datetime.now().year
            if date.year > current_year + 9:
                year = str(date.year)[-2:]  # Last 2 digits (e.g., 35 for 2035)
            else:
                year = str(date.year)[-1]   # Last 1 digit (e.g., 5 for 2025)
            return f"{imm_dates[date.month]}{year}"
        return None
    
    def get_effective_bucket(self, start_date, today, currency, mpc_dates, imm_dates):
        """Convert start date to market convention with priority order:
        1. MPC dates (BOE1, FED1, etc.)
        2. IMM dates (H6, M6, U6, Z6)
        3. Standard tenors (1Y, 6M, 9M, etc.)
        4. Fallback to original date
        """
        # 1. FIRST PRIORITY: Check for MPC dates
        mpc_name = self.get_mpc_name(start_date, currency, mpc_dates)
        if mpc_name:
            return mpc_name
        
        delta_days = (start_date - today).days
        
        # 2. SECOND PRIORITY: Spot (within 5 days)
        if abs(delta_days) <= 5:
            return "Spot"
        
        # 2.5. Check for weekly periods (1W, 2W, 3W, 4W)
        if abs(delta_days - 7) <= 1:
            return "1W"
        if abs(delta_days - 14) <= 1:
            return "2W"
        if abs(delta_days - 21) <= 1:
            return "3W"
        if abs(delta_days - 28) <= 1:
            return "4W"
        
        # 3. THIRD PRIORITY: Check for IMM dates (3rd Wednesday of Mar/Jun/Sep/Dec)
        imm_code = self.get_imm_code(start_date, imm_dates)
        if imm_code:
            return imm_code
        
        # 4. FOURTH PRIORITY: Check for standard periods (6M, 9M, 1Y)
        months_delta = round(delta_days / 30.4375)  # average days per month
        if abs(months_delta - 6) <= 1:
            return "6M"
        if abs(months_delta - 9) <= 1:
            return "9M"
        if abs(months_delta - 12) <= 1:
            return "1Y"
        
        # 5. FIFTH PRIORITY: Check for yearly periods (1Y, 2Y, 3Y, etc.)
        rel_years = delta_days / 365.25
        for y in range(1, 11):
            if abs(rel_years - y) <= 0.2:
                return f"{y}Y"
        
        # 6. FALLBACK: Original date format
        return start_date.strftime('%Y-%m-%d')
    
    def get_expiration_bucket(self, expiration_date, today, currency, mpc_dates, imm_dates):
        """Convert expiration date to market convention with priority order:
        1. MPC dates (BOE1, FED1, etc.)
        2. IMM dates (H6, M6, U6, Z6)
        3. Standard tenors (1Y, 6M, 9M, etc.)
        4. Fallback to original date
        """
        if pd.isna(expiration_date):
            return ""
        
        # 1. FIRST PRIORITY: Check for MPC dates
        mpc_name = self.get_mpc_name(expiration_date, currency, mpc_dates)
        if mpc_name:
            return mpc_name
        
        delta_days = (expiration_date - today).days
        
        # 2. SECOND PRIORITY: Spot (within 5 days)
        if abs(delta_days) <= 5:
            return "Spot"
        
        # 2.5. Check for weekly periods (1W, 2W, 3W, 4W)
        if abs(delta_days - 7) <= 1:
            return "1W"
        if abs(delta_days - 14) <= 1:
            return "2W"
        if abs(delta_days - 21) <= 1:
            return "3W"
        if abs(delta_days - 28) <= 1:
            return "4W"
        
        # 3. THIRD PRIORITY: Check for IMM dates (3rd Wednesday of Mar/Jun/Sep/Dec)
        imm_code = self.get_imm_code(expiration_date, imm_dates)
        if imm_code:
            return imm_code
        
        # 4. FOURTH PRIORITY: Check for standard periods (6M, 9M, 1Y)
        months_delta = round(delta_days / 30.4375)  # average days per month
        if abs(months_delta - 6) <= 1:
            return "6M"
        if abs(months_delta - 9) <= 1:
            return "9M"
        if abs(months_delta - 12) <= 1:
            return "1Y"
        
        # 5. FIFTH PRIORITY: Check for yearly periods (1Y, 2Y, 3Y, etc.)
        rel_years = delta_days / 365.25
        for y in range(1, 11):
            if abs(rel_years - y) <= 0.2:
                return f"{y}Y"
        
        # 6. FALLBACK: Original date format
        return expiration_date.strftime('%Y-%m-%d')

    def process_trades(self, trade_list: List[Dict], source: str) -> List[Dict]:
        """Process raw trade data into structured format"""
        processed_data = []
        current_time = datetime.now()
        today = datetime.now().date()
        
        # Load MPC, IMM dates, and FX rates once for this batch
        mpc_dates = self.load_mpc_dates()
        imm_dates = self.load_imm_dates()
        fx_rates = self.load_fx_rates()
        
        for trade in trade_list:
            try:
                # Extract basic fields
                trade_time = trade.get('eventTimestamp', '')
                effective_date = trade.get('effectiveDate', '')
                expiration_date = trade.get('expirationDate', '')
                currency = trade.get('notionalCurrencyLeg1', '')
                rates = trade.get('fixedRateLeg1', '') or trade.get('spreadLeg1', '')
                notionals = trade.get('notionalAmountLeg1', '')
                action_type = trade.get('actionType', '')
                event_type = trade.get('eventType', '')
                asset_class = trade.get('assetClass', '')
                upi_underlier_name = trade.get('uniqueProductIdentifierUnderlierName', '')
                unique_product_identifier = trade.get('uniqueProductIdentifier', '')
                dissemination_identifier = trade.get('disseminationIdentifier', '')
                original_dissemination_identifier = trade.get('originalDisseminationIdentifier', '')
                
                # Additional fields
                frequency = trade.get('Settlement currency-Leg 1', '')
                other_payment_type = trade.get('otherPaymentType', '')
                package_indicator = trade.get('packageIndicator', '')
                floating_rate_payment_frequency_period_leg2 = trade.get('floatingRatePaymentFrequencyPeriodLeg2', '')
                floating_rate_payment_frequency_period_multiplier_leg2 = trade.get('floatingRatePaymentFrequencyPeriodMultiplierLeg2', '')
                fixed_rate_payment_frequency_period_leg1 = trade.get('fixedRatePaymentFrequencyPeriodLeg1', '')
                fixed_rate_payment_frequency_period_multiplier_leg1 = trade.get('fixedRatePaymentFrequencyPeriodMultiplierLeg1', '')
                
                # Calculate tenor in years and DV01
                tenor_in_years = None
                effective_bucket = ""
                expiration_bucket = ""
                dv01 = 0.0
                
                if effective_date and expiration_date:
                    try:
                        effective_dt = datetime.strptime(effective_date, '%Y-%m-%d').date()
                        expiration_dt = datetime.strptime(expiration_date, '%Y-%m-%d').date()
                        
                        # Skip trades with effective date before today (no past dated trades)
                        if effective_dt < today:
                            logger.info(f"Skipping past dated trade: {effective_date} < {today}")
                            continue
                        
                        tenor_in_years = (expiration_dt - effective_dt).days / 365.25
                        
                        # Calculate effective and expiration buckets using the same logic as DTCCAnalysis
                        effective_bucket = self.get_effective_bucket(effective_dt, today, currency, mpc_dates, imm_dates)
                        expiration_bucket = self.get_expiration_bucket(expiration_dt, today, currency, mpc_dates, imm_dates)
                        
                        # Calculate DV01 with proper swap formula and FX conversion
                        if rates and notionals and currency:
                            try:
                                # Clean and convert numeric values
                                rates_val = float(str(rates).replace(',', '').replace('+', '')) if rates else 0.0
                                notional_val = float(str(notionals).replace(',', '').replace('+', '')) if notionals else 0.0
                                
                                if rates_val != 0.0 and notional_val != 0.0:
                                    effective_dt_datetime = datetime.strptime(effective_date, '%Y-%m-%d')
                                    expiration_dt_datetime = datetime.strptime(expiration_date, '%Y-%m-%d')
                                    
                                    # Calculate DV01 in local currency
                                    dv01_local = self.calculate_dv01(
                                        notional=notional_val, 
                                        rate=rates_val,
                                        effective_date=effective_dt_datetime, 
                                        expiration_date=expiration_dt_datetime,
                                        frequency=frequency or 'Semi-Annual'
                                    )
                                    
                                    # Convert to USD
                                    dv01 = self.convert_dv01_to_usd(dv01_local, currency, fx_rates)
                                    
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Error calculating DV01: {e}")
                                dv01 = 0.0
                        
                    except ValueError:
                        pass
                
                processed_trade = {
                    'Source': source,
                    'Trade Time': trade_time,
                    'Effective Date': effective_date,
                    'Expiration Date': expiration_date,
                    'Effective Bucket': effective_bucket,
                    'Expiration Bucket': expiration_bucket,
                    'Tenor': tenor_in_years,
                    'Currency': currency,
                    'Rates': rates,
                    'Notionals': notionals,
                    'DV01': dv01,
                    'Frequency': frequency,
                    'Action Type': action_type,
                    'Event Type': event_type,
                    'Asset Class': asset_class,
                    'UPI Underlier Name': upi_underlier_name,
                    'Unique Product Identifier': unique_product_identifier,
                    'Dissemination Identifier': dissemination_identifier,
                    'Original Dissemination Identifier': original_dissemination_identifier,
                    'Other Payment Type': other_payment_type,
                    'Package Indicator': package_indicator,
                    'Floating Rate Payment Frequency Period Leg2': floating_rate_payment_frequency_period_leg2,
                    'Floating Rate Payment Frequency Period Multiplier Leg2': floating_rate_payment_frequency_period_multiplier_leg2,
                    'Fixed Rate Payment Frequency Period Leg1': fixed_rate_payment_frequency_period_leg1,
                    'Fixed Rate Payment Frequency Period Multiplier Leg1': fixed_rate_payment_frequency_period_multiplier_leg1,
                    'Created At': current_time.isoformat(),
                    'Updated At': current_time.isoformat()
                }
                
                processed_data.append(processed_trade)
                
            except Exception as e:
                logger.warning(f"Error processing trade: {e}")
                continue
        
        return processed_data
    
    def handle_trade_modifications(self, processed_trades: List[Dict]) -> List[Dict]:
        """Handle trade modifications and duplicates"""
        trades_to_add = []
        trades_to_remove = []
        
        for trade in processed_trades:
            dissemination_id = trade.get('Dissemination Identifier', '')
            original_dissemination_id = trade.get('Original Dissemination Identifier', '')
            
            # Skip if no dissemination identifier
            if not dissemination_id:
                logger.warning("Trade without dissemination identifier skipped")
                continue
            
            # Check if this is a modification (has Original Dissemination Identifier)
            if original_dissemination_id:
                # This is a modified trade - remove the old trade
                if original_dissemination_id in self.existing_dissemination_ids:
                    trades_to_remove.append(original_dissemination_id)
                    logger.info(f"Trade modification detected: {original_dissemination_id} -> {dissemination_id}")
            
            # Check for duplicates
            if dissemination_id not in self.existing_dissemination_ids:
                trades_to_add.append(trade)
                # Update tracking sets
                self.existing_dissemination_ids.add(dissemination_id)
                if original_dissemination_id:
                    self.existing_original_dissemination_ids.add(original_dissemination_id)
            else:
                logger.info(f"Duplicate trade skipped: {dissemination_id}")
        
        # Remove modified trades from CSV
        if trades_to_remove:
            self._remove_trades_from_csv(trades_to_remove)
        
        return trades_to_add
    
    def _remove_trades_from_csv(self, dissemination_ids_to_remove: List[str]):
        """Remove trades with specified dissemination IDs from CSV"""
        try:
            # Read all rows
            rows = []
            with open(self.csv_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
            
            # Filter out rows to remove
            filtered_rows = [
                row for row in rows 
                if row.get('Dissemination Identifier', '') not in dissemination_ids_to_remove
            ]
            
            # Write back filtered rows
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(filtered_rows)
            
            removed_count = len(rows) - len(filtered_rows)
            logger.info(f"Removed {removed_count} modified trades from CSV")
            
            # Update tracking sets
            for id_to_remove in dissemination_ids_to_remove:
                self.existing_dissemination_ids.discard(id_to_remove)
                self.existing_original_dissemination_ids.discard(id_to_remove)
            
        except Exception as e:
            logger.error(f"Error removing trades from CSV: {e}")
    
    def append_to_csv(self, new_trades: List[Dict]):
        """Append new trades to CSV file"""
        if not new_trades:
            return
        
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerows(new_trades)
            
            logger.info(f"Added {len(new_trades)} new trades to {self.csv_file}")
            
        except Exception as e:
            logger.error(f"Error writing to CSV file: {e}")
    
    def run_fetch_cycle(self):
        """Run one complete fetch cycle for all sources"""
        try:
            logger.info("Starting DTCC data fetch cycle for all sources...")
            
            # Fetch data from all sources
            all_trades_by_source = self.fetch_all_sources()
            
            total_processed = 0
            total_new_trades = 0
            
            # Process each source
            for source, trades in all_trades_by_source.items():
                if not trades:
                    logger.info(f"No trades from {source}")
                    continue
                
                logger.info(f"Processing {len(trades)} trades from {source}")
                
                # Process trades for this source
                processed_trades = self.process_trades(trades, source)
                logger.info(f"Processed {len(processed_trades)} trades from {source}")
                total_processed += len(processed_trades)
                
                # Handle modifications and duplicates
                new_trades = self.handle_trade_modifications(processed_trades)
                
                # Add new trades to CSV
                if new_trades:
                    self.append_to_csv(new_trades)
                    logger.info(f"Added {len(new_trades)} new trades from {source} to CSV")
                    total_new_trades += len(new_trades)
                else:
                    logger.info(f"No new trades from {source}")
            
            logger.info(f"Fetch cycle completed: {total_processed} total processed, {total_new_trades} new trades added")
            
        except Exception as e:
            logger.error(f"Error in fetch cycle: {e}")
    
    def start(self):
        """Start the continuous fetching process"""
        if self.running:
            logger.warning("Fetcher is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info(f"DTCC Fetcher started - running every {FETCH_INTERVAL} seconds")
    
    def stop(self):
        """Stop the fetching process"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("DTCC Fetcher stopped")
    
    def _run_loop(self):
        """Main loop for continuous fetching"""
        while self.running:
            try:
                self.run_fetch_cycle()
                
                # Wait for next cycle
                for _ in range(FETCH_INTERVAL):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def run_once(self):
        """Run a single fetch cycle"""
        logger.info("Running single fetch cycle...")
        self.run_fetch_cycle()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    if 'fetcher' in globals():
        fetcher.stop()
    sys.exit(0)

def main():
    """Main function"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create fetcher instance
    fetcher = DTCCFetcher()
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--once':
            # Run once and exit
            fetcher.run_once()
        else:
            # Run continuously
            fetcher.start()
            
            # Keep main thread alive
            while fetcher.running:
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        fetcher.stop()
        logger.info("DTCC Fetcher shutdown complete")

if __name__ == "__main__":
    main()

