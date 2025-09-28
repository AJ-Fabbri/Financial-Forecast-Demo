#!/usr/bin/env python3
"""
Database management script for Financial Forecast Demo
Handles populating and managing the DuckDB database
"""
import argparse
import sys
from datetime import datetime

from ..data.database import get_database, update_ticker_from_yfinance
from ..data.loader import populate_database_with_tickers
from ..config import DEFAULT_TICKER


def list_tickers():
    """List all tickers in the database"""
    print("Listing tickers in database...")
    db = get_database()
    tickers = db.list_tickers()
    db.close()
    
    if tickers:
        print(f"\nFound {len(tickers)} tickers:")
        for ticker in tickers:
            print(f"  - {ticker}")
    else:
        print("No tickers found in database")
    
    return tickers


def show_ticker_info(ticker):
    """Show information about a specific ticker"""
    print(f"Getting info for {ticker}...")
    db = get_database()
    info = db.get_ticker_info(ticker)
    db.close()
    
    if 'error' in info:
        print(f"Error: {info['error']}")
        return
    
    if info['row_count'] == 0:
        print(f"No data found for {ticker}")
        return
    
    print(f"\nTicker: {info['ticker']}")
    print(f"Records: {info['row_count']}")
    print(f"Date range: {info['start_date']} to {info['end_date']}")


def update_ticker(ticker, years=5):
    """Update a single ticker"""
    print(f"Updating {ticker} with {years} years of data...")
    success = update_ticker_from_yfinance(ticker, years)
    
    if success:
        print(f" Successfully updated {ticker}")
        show_ticker_info(ticker)
    else:
        print(f" Failed to update {ticker}")
    
    return success


def update_multiple_tickers(tickers, years=5):
    """Update multiple tickers"""
    print(f"Updating {len(tickers)} tickers with {years} years of data...")
    results = populate_database_with_tickers(tickers, years)
    
    # Summary
    successful = sum(1 for r in results.values() if r['success'])
    failed = len(results) - successful
    
    print(f"\n Update Summary:")
    print(f" Successful: {successful}")
    print(f" Failed: {failed}")
    
    if failed > 0:
        print("\nFailed tickers:")
        for ticker, result in results.items():
            if not result['success']:
                error = result.get('error', 'Unknown error')
                print(f"  - {ticker}: {error}")
    
    return results


def populate_demo_data():
    """Populate database with common demo tickers"""
    demo_tickers = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    
    print(" Populating database with demo data...")
    print(f"Tickers: {', '.join(demo_tickers)}")
    
    results = update_multiple_tickers(demo_tickers, years=5)
    
    print("\n Demo data population complete!")
    return results


def clear_ticker(ticker):
    """Remove a ticker from the database"""
    print(f"Removing {ticker} from database...")
    try:
        db = get_database()
        db.conn.execute("DELETE FROM prices WHERE ticker = ?", [ticker])
        rows_affected = db.conn.execute("SELECT changes()").fetchone()[0]
        db.close()
        
        if rows_affected > 0:
            print(f" Removed {rows_affected} rows for {ticker}")
        else:
            print(f"No data found for {ticker}")
            
    except Exception as e:
        print(f" Error removing {ticker}: {e}")


def database_stats():
    """Show database statistics"""
    print(" Database Statistics")
    print("=" * 40)
    
    try:
        db = get_database()
        
        # Overall stats
        total_rows = db.conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        total_tickers = db.conn.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
        
        print(f"Total records: {total_rows:,}")
        print(f"Total tickers: {total_tickers}")
        
        if total_rows > 0:
            # Date range
            date_range = db.conn.execute(
                "SELECT MIN(date) as start_date, MAX(date) as end_date FROM prices"
            ).fetchone()
            print(f"Date range: {date_range[0]} to {date_range[1]}")
            
            # Top tickers by record count
            print("\nTop tickers by record count:")
            top_tickers = db.conn.execute("""
                SELECT ticker, COUNT(*) as count 
                FROM prices 
                GROUP BY ticker 
                ORDER BY count DESC 
                LIMIT 10
            """).fetchall()
            
            for ticker, count in top_tickers:
                print(f"  {ticker}: {count:,} records")
        
        db.close()
        
    except Exception as e:
        print(f"Error getting database stats: {e}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Manage Financial Forecast Demo database',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    subparsers.add_parser('list', help='List all tickers in database')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show info for a ticker')
    info_parser.add_argument('ticker', help='Ticker symbol')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update ticker data')
    update_parser.add_argument('ticker', help='Ticker symbol')
    update_parser.add_argument('--years', type=int, default=5, 
                              help='Years of historical data (default: 5)')
    
    # Update multiple command
    multi_parser = subparsers.add_parser('update-multiple', help='Update multiple tickers')
    multi_parser.add_argument('tickers', nargs='+', help='Ticker symbols')
    multi_parser.add_argument('--years', type=int, default=5,
                             help='Years of historical data (default: 5)')
    
    # Populate demo data
    subparsers.add_parser('populate-demo', help='Populate with demo tickers')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Remove ticker from database')
    clear_parser.add_argument('ticker', help='Ticker symbol to remove')
    
    # Stats command
    subparsers.add_parser('stats', help='Show database statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'list':
            list_tickers()
            
        elif args.command == 'info':
            show_ticker_info(args.ticker.upper())
            
        elif args.command == 'update':
            update_ticker(args.ticker.upper(), args.years)
            
        elif args.command == 'update-multiple':
            tickers = [t.upper() for t in args.tickers]
            update_multiple_tickers(tickers, args.years)
            
        elif args.command == 'populate-demo':
            populate_demo_data()
            
        elif args.command == 'clear':
            clear_ticker(args.ticker.upper())
            
        elif args.command == 'stats':
            database_stats()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 