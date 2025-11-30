import sqlite3
import os
from datetime import datetime

class Database:
    def __init__(self, db_path="market_prices.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT NOT NULL,
            crop TEXT NOT NULL,
            date DATE NOT NULL,
            current_price REAL,
            price_30 REAL,
            price_60 REAL,
            price_90 REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS crops_by_market (
            market TEXT PRIMARY KEY,
            crops TEXT
        )
        """)
        
        conn.commit()
        conn.close()
        print("✓ Database initialized")
    
    def insert_prediction(self, market, crop, current_price, price_30, price_60, price_90):
        """Insert price prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO market_data (market, crop, date, current_price, price_30, price_60, price_90)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (market, crop, datetime.now().date(), current_price, price_30, price_60, price_90))
        
        conn.commit()
        conn.close()
    
    def get_latest_prices(self, market, crop):
        """Get latest predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT current_price, price_30, price_60, price_90 
        FROM market_data 
        WHERE market = ? AND crop = ? 
        ORDER BY date DESC 
        LIMIT 1
        """, (market, crop))
        
        result = cursor.fetchone()
        conn.close()
        
        return result

