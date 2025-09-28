"""
Database operations for Financial Forecast Demo using DuckDB
Simplified version of the original database functionality
"""
import duckdb
import pandas as pd
import os
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime
import yfinance as yf

from ..config import PROJECT_ROOT

# Database file location
DB_PATH = os.path.join(PROJECT_ROOT, "delphi_demo.duckdb")


class FinancialForecastDatabase:
    """
    Simplified database class for managing financial data in DuckDB
    Adapted for demonstration purposes
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize database connection
        
        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Connect to DuckDB database"""
        try:
            self.conn = duckdb.connect(self.db_path)
            print(f"Connected to DuckDB database at: {self.db_path}")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise
    
    def _create_tables(self):
        """Create the prices table if it doesn't exist"""
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS prices (
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                dividends DOUBLE DEFAULT 0.0,
                stock_splits DOUBLE DEFAULT 0.0,
                PRIMARY KEY (ticker, date)
            )
            """
            self.conn.execute(create_table_sql)
            print("Prices table ready")
        except Exception as e:
            print(f"Error creating tables: {e}")
            raise
    
    def upsert_ticker_data(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Insert or update ticker data in the database
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data (index should be dates)
            
        Returns:
            Number of rows affected
        """
        if df.empty:
            print(f"No data to insert for {ticker}")
            return 0
        
        try:
            # Prepare the DataFrame
            df_to_insert = df.copy()
            df_to_insert['ticker'] = ticker
            df_to_insert['date'] = df_to_insert.index
            df_to_insert = df_to_insert.reset_index(drop=True)
            
            # Rename columns to match database schema
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            }
            
            df_to_insert = df_to_insert.rename(columns=column_mapping)
            
            # Select only columns that exist in the database
            db_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            df_to_insert = df_to_insert[[col for col in db_columns if col in df_to_insert.columns]]
            
            # Insert or replace data
            self.conn.execute("DELETE FROM prices WHERE ticker = ?", [ticker])
            self.conn.execute("INSERT INTO prices SELECT * FROM df_to_insert")
            
            rows_affected = len(df_to_insert)
            print(f"Upserted {rows_affected} rows for {ticker}")
            return rows_affected
            
        except Exception as e:
            print(f"Error upserting data for {ticker}: {e}")
            raise
    
    def get_ticker_data(self, ticker: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve ticker data from database
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            DataFrame with ticker data
        """
        try:
            query = "SELECT * FROM prices WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            df = self.conn.execute(query, params).df()
            
            if df.empty:
                print(f"No data found for {ticker} in database")
                return pd.DataFrame()
            
            # Format the DataFrame
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.drop('ticker', axis=1)
            
            # Capitalize column names to match convention
            df.columns = df.columns.str.title()
            df = df.rename(columns={'Stock_Splits': 'Stock Splits'})
            
            return df
            
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")
            return pd.DataFrame()
    
    def list_tickers(self) -> List[str]:
        """Get list of all tickers in database"""
        try:
            result = self.conn.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchall()
            return [row[0] for row in result]
        except Exception as e:
            print(f"Error listing tickers: {e}")
            return []
    
    def get_ticker_info(self, ticker: str) -> dict:
        """Get information about a ticker in the database"""
        try:
            query = """
            SELECT 
                COUNT(*) as row_count,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM prices 
            WHERE ticker = ?
            """
            result = self.conn.execute(query, [ticker]).fetchone()
            
            if result and result[0] > 0:
                return {
                    'ticker': ticker,
                    'row_count': result[0],
                    'start_date': result[1],
                    'end_date': result[2]
                }
            else:
                return {'ticker': ticker, 'row_count': 0}
                
        except Exception as e:
            print(f"Error getting ticker info for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")


def get_database() -> FinancialForecastDatabase:
    """Get a database instance (singleton pattern)"""
    return FinancialForecastDatabase()


def update_ticker_from_yfinance(ticker: str, years: int = 5) -> bool:
    """
    Fetch ticker data from yfinance and update database
    
    Args:
        ticker: Stock ticker symbol
        years: Number of years of historical data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Fetching {ticker} data from yfinance...")
        
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{years}y", auto_adjust=True)
        
        if df.empty:
            print(f"No data available for {ticker}")
            return False
        
        # Update database
        db = get_database()
        rows_affected = db.upsert_ticker_data(ticker, df)
        db.close()
        
        print(f"Successfully updated {ticker}: {rows_affected} rows")
        return True
        
    except Exception as e:
        print(f"Error updating {ticker}: {e}")
        return False 