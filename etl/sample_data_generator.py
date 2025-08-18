#!/usr/bin/env python
"""
Sample data generator for Card Analytics Platform
Generates realistic card transaction data for testing
"""

import random
import hashlib
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from faker import Faker
import click
from loguru import logger

# Initialize Faker for different locales
fake_uz = Faker('ru_RU')  # Using Russian locale as proxy for Uzbek
fake_en = Faker('en_US')
Faker.seed(42)
random.seed(42)
np.random.seed(42)


# Constants for data generation
UZBEK_REGIONS = [
    'Tashkent', 'Samarkand', 'Bukhara', 'Andijan', 'Fergana',
    'Namangan', 'Qashqadaryo', 'Surxondaryo', 'Xorazm', 'Navoiy',
    'Jizzax', 'Sirdaryo', 'Tashkent Region', 'Karakalpakstan'
]

UZBEK_BANKS = [
    'Uzpromstroybank', 'Asaka Bank', 'Xalq Bank', 'Qishloq Qurilish Bank',
    'Ipoteka Bank', 'Turon Bank', 'Hamkor Bank', 'Kapital Bank',
    'Aloqa Bank', 'Ipak Yuli Bank', 'Orient Finans Bank', 'Ravnaq Bank',
    'Tenge Bank', 'Universal Bank', 'Ziraat Bank', 'KDB Bank'
]

CARD_TYPES = ['debit', 'credit', 'prepaid']
CARD_PRODUCTS = ['Visa Classic', 'Visa Gold', 'Visa Platinum', 'MasterCard Standard', 
                  'MasterCard Gold', 'MasterCard Platinum', 'UzCard', 'Humo', 
                  'UnionPay', 'Mir']

# MCC codes with categories and typical amounts
MCC_CATEGORIES = {
    5411: ('Grocery Stores', 50000, 500000),
    5541: ('Service Stations', 100000, 800000),
    5812: ('Restaurants', 30000, 300000),
    5912: ('Drug Stores', 20000, 200000),
    5311: ('Department Stores', 100000, 1000000),
    5732: ('Electronics Stores', 200000, 5000000),
    5999: ('Miscellaneous Retail', 50000, 500000),
    6011: ('ATM Cash Withdrawal', 100000, 2000000),
    4111: ('Local Transport', 10000, 50000),
    4121: ('Taxi Services', 20000, 150000),
    5814: ('Fast Food', 15000, 100000),
    5813: ('Bars & Nightclubs', 50000, 500000),
    5251: ('Hardware Stores', 50000, 800000),
    5942: ('Book Stores', 20000, 200000),
    5661: ('Shoe Stores', 100000, 800000),
    5691: ('Clothing Stores', 100000, 1500000),
    7230: ('Beauty Shops', 50000, 300000),
    8211: ('Schools', 500000, 5000000),
    8062: ('Hospitals', 100000, 2000000),
    6012: ('Financial Institutions', 100000, 10000000)
}

TERMINAL_TYPES = ['ATM', 'POS', 'ECOM', 'MOTO', 'IMPRINTER']

P2P_TYPES = ['wallet_transfer', 'card_transfer', 'mobile_transfer', 'bank_transfer']


def generate_hpan() -> str:
    """Generate hashed PAN"""
    pan = ''.join([str(random.randint(0, 9)) for _ in range(16)])
    return hashlib.sha256(pan.encode()).hexdigest()[:32]


def generate_pinfl() -> str:
    """Generate personal identification number"""
    return ''.join([str(random.randint(0, 9)) for _ in range(14)])


def generate_transaction_code() -> str:
    """Generate unique transaction code"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    random_part = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    return f"TXN{timestamp}{random_part}"


def generate_merchant_name(mcc: int) -> str:
    """Generate merchant name based on MCC"""
    category = MCC_CATEGORIES.get(mcc, ('Unknown', 0, 0))[0]
    
    merchant_templates = {
        'Grocery Stores': ['MaxWay', 'Korzinka', 'Makro', 'Havas', 'GrossMarket'],
        'Service Stations': ['UzPetrol', 'Shel', 'Lukoil', 'Gazprom', 'UNG'],
        'Restaurants': ['Afsona', 'Caravan', 'Jumanji', 'Plov Center', 'Bella Italia'],
        'Drug Stores': ['Dori-Darmon', 'Apteka Plus', 'Med Pharm', '36.6', 'Evalar'],
        'Fast Food': ['Evos', 'KFC', 'MaxWay', 'Street 77', 'Bellissimo'],
        'Electronics Stores': ['MediaPark', 'TechnoMart', 'Samsung Store', 'Mi Store'],
    }
    
    base_category = category.replace(' & ', ' ').split()[0] + ' ' + category.replace(' & ', ' ').split()[-1]
    names = merchant_templates.get(category, [f"{base_category} {i}" for i in range(1, 6)])
    return random.choice(names) + f" #{random.randint(1, 999)}"


def generate_oked() -> str:
    """Generate OKED (economic activity classifier)"""
    return f"{random.randint(10, 99)}{random.randint(10, 99)}{random.randint(0, 9)}"


def generate_sample_transactions(
    num_records: int = 10000,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    fraud_rate: float = 0.01,
    p2p_rate: float = 0.2
) -> pd.DataFrame:
    """
    Generate sample transaction data
    
    Args:
        num_records: Number of records to generate
        start_date: Start date for transactions
        end_date: End date for transactions
        fraud_rate: Percentage of fraudulent transactions
        p2p_rate: Percentage of P2P transactions
    
    Returns:
        DataFrame with generated transactions
    """
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)
    if end_date is None:
        end_date = datetime.now()
    
    logger.info(f"Generating {num_records} transactions from {start_date} to {end_date}")
    
    # Generate unique cards and customers
    num_cards = max(100, num_records // 20)
    num_customers = max(50, num_cards // 2)
    
    # Pre-generate cards
    cards = []
    customers = []
    
    for i in range(num_customers):
        pinfl = generate_pinfl()
        gender = random.choice(['M', 'F'])
        birth_year = random.randint(1950, 2005)
        customers.append({
            'pinfl': pinfl,
            'gender': gender,
            'birth_year': birth_year,
            'age': 2024 - birth_year,
            'region': random.choice(UZBEK_REGIONS)
        })
    
    for i in range(num_cards):
        customer = random.choice(customers)
        issue_date = start_date - timedelta(days=random.randint(30, 1095))
        expire_date = issue_date + timedelta(days=1095)  # 3 years
        
        cards.append({
            'hpan': generate_hpan(),
            'pinfl': customer['pinfl'],
            'gender': customer['gender'],
            'birth_year': customer['birth_year'],
            'age': customer['age'],
            'emitent_region': customer['region'],
            'card_type': random.choice(CARD_TYPES),
            'card_product_type': random.choice(CARD_PRODUCTS),
            'issue_date': issue_date,
            'expire_date': expire_date,
            'emitent_bank': random.choice(UZBEK_BANKS),
            'reissuing_flag': random.choice([0, 0, 0, 0, 1])  # 20% reissued
        })
    
    # Generate transactions
    transactions = []
    
    for i in range(num_records):
        # Select random card
        card = random.choice(cards)
        
        # Generate transaction datetime
        tx_datetime = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        # Determine if P2P
        is_p2p = random.random() < p2p_rate
        
        if is_p2p:
            # P2P transaction
            sender_card = card
            receiver_card = random.choice([c for c in cards if c['hpan'] != sender_card['hpan']])
            
            transaction = {
                'transaction_code': generate_transaction_code(),
                'rday': tx_datetime.date(),
                'sttl_date': (tx_datetime + timedelta(days=random.randint(0, 2))).date(),
                'hour': tx_datetime.hour,
                'minute': tx_datetime.minute,
                
                # Card info
                'hpan': sender_card['hpan'],
                'card_product_type': sender_card['card_product_type'],
                'card_type': sender_card['card_type'],
                'expire_date': sender_card['expire_date'].date(),
                'issue_date': sender_card['issue_date'].date(),
                'reissuing_flag': sender_card['reissuing_flag'],
                
                # Customer info
                'pinfl': sender_card['pinfl'],
                'gender': sender_card['gender'],
                'birth_year': sender_card['birth_year'],
                'age': sender_card['age'],
                'age_group': f"{(sender_card['age'] // 10) * 10}-{(sender_card['age'] // 10) * 10 + 9}",
                'emitent_region': sender_card['emitent_region'],
                
                # Transaction amounts
                'amount_uzs': round(random.uniform(10000, 5000000), 2),
                'reqamt': 0,
                'conamt': 0,
                
                # Transaction details
                'credit_debit': 'debit',
                'reversal_flag': 0 if random.random() > 0.01 else 1,
                'respcode': '00' if random.random() > 0.05 else random.choice(['05', '51', '61']),
                'refnum': f"REF{random.randint(100000000, 999999999)}",
                'fe_refnum': f"FER{random.randint(100000000, 999999999)}",
                'fe_stan': str(random.randint(100000, 999999)),
                'fe_trace': str(random.randint(100000, 999999)),
                
                # Merchant info (empty for P2P)
                'mcc': 6012,  # Financial institution
                'merchant_name': 'P2P Transfer',
                'merchant_type': 'P2P',
                'oked': '',
                'terminal_type': 'ECOM',
                'terminal_id': f"TERM{random.randint(10000000, 99999999)}",
                
                # Banks
                'emitent_bank': sender_card['emitent_bank'],
                'acquirer_bank': receiver_card['emitent_bank'],
                
                # P2P specific
                'p2p_flag': 1,
                'p2p_type': random.choice(P2P_TYPES),
                'sender_hpan': sender_card['hpan'],
                'sender_bank': sender_card['emitent_bank'],
                'receiver_hpan': receiver_card['hpan'],
                'receiver_bank': receiver_card['emitent_bank']
            }
        else:
            # Regular transaction
            mcc = random.choice(list(MCC_CATEGORIES.keys()))
            category, min_amount, max_amount = MCC_CATEGORIES[mcc]
            
            transaction = {
                'transaction_code': generate_transaction_code(),
                'rday': tx_datetime.date(),
                'sttl_date': (tx_datetime + timedelta(days=random.randint(0, 2))).date(),
                'hour': tx_datetime.hour,
                'minute': tx_datetime.minute,
                
                # Card info
                'hpan': card['hpan'],
                'card_product_type': card['card_product_type'],
                'card_type': card['card_type'],
                'expire_date': card['expire_date'].date(),
                'issue_date': card['issue_date'].date(),
                'reissuing_flag': card['reissuing_flag'],
                
                # Customer info
                'pinfl': card['pinfl'],
                'gender': card['gender'],
                'birth_year': card['birth_year'],
                'age': card['age'],
                'age_group': f"{(card['age'] // 10) * 10}-{(card['age'] // 10) * 10 + 9}",
                'emitent_region': card['emitent_region'],
                
                # Transaction amounts
                'amount_uzs': round(random.uniform(min_amount, max_amount), 2),
                'reqamt': 0,
                'conamt': 0,
                
                # Transaction details
                'credit_debit': 'debit' if random.random() > 0.1 else 'credit',
                'reversal_flag': 0 if random.random() > 0.01 else 1,
                'respcode': '00' if random.random() > 0.05 else random.choice(['05', '51', '61']),
                'refnum': f"REF{random.randint(100000000, 999999999)}",
                'fe_refnum': f"FER{random.randint(100000000, 999999999)}",
                'fe_stan': str(random.randint(100000, 999999)),
                'fe_trace': str(random.randint(100000, 999999)),
                
                # Merchant info
                'mcc': mcc,
                'merchant_name': generate_merchant_name(mcc),
                'merchant_type': category,
                'oked': generate_oked(),
                'terminal_type': random.choice(TERMINAL_TYPES),
                'terminal_id': f"TERM{random.randint(10000000, 99999999)}",
                
                # Banks
                'emitent_bank': card['emitent_bank'],
                'acquirer_bank': random.choice(UZBEK_BANKS),
                
                # P2P specific (empty for regular transactions)
                'p2p_flag': 0,
                'p2p_type': '',
                'sender_hpan': '',
                'sender_bank': '',
                'receiver_hpan': '',
                'receiver_bank': ''
            }
        
        # Add potential fraud indicators
        if random.random() < fraud_rate:
            # Make transaction suspicious
            transaction['amount_uzs'] *= random.uniform(5, 20)  # Unusually high amount
            transaction['hour'] = random.choice([2, 3, 4, 23])  # Odd hours
            transaction['respcode'] = random.choice(['05', '51', '61', '91'])  # Declined
        
        transactions.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Sort by date
    df = df.sort_values('rday').reset_index(drop=True)
    
    logger.info(f"Generated {len(df)} transactions")
    logger.info(f"Date range: {df['rday'].min()} to {df['rday'].max()}")
    logger.info(f"P2P transactions: {df['p2p_flag'].sum()} ({df['p2p_flag'].mean()*100:.1f}%)")
    logger.info(f"Unique cards: {df['hpan'].nunique()}")
    logger.info(f"Unique merchants: {df['merchant_name'].nunique()}")
    
    return df


@click.command()
@click.option('--records', '-n', default=10000, help='Number of records to generate')
@click.option('--output', '-o', default='data/samples/sample_transactions.csv', 
              help='Output file path')
@click.option('--start-date', '-s', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', help='End date (YYYY-MM-DD)')
@click.option('--fraud-rate', '-f', default=0.01, help='Fraud rate (0-1)')
@click.option('--p2p-rate', '-p', default=0.2, help='P2P transaction rate (0-1)')
def main(records, output, start_date, end_date, fraud_rate, p2p_rate):
    """Generate sample transaction data"""
    
    # Parse dates
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate data
    df = generate_sample_transactions(
        num_records=records,
        start_date=start_date,
        end_date=end_date,
        fraud_rate=fraud_rate,
        p2p_rate=p2p_rate
    )
    
    # Save to file
    df.to_csv(output, index=False)
    logger.info(f"Data saved to {output}")
    
    # Show summary
    print("\n" + "="*50)
    print("SAMPLE DATA SUMMARY")
    print("="*50)
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['rday'].min()} to {df['rday'].max()}")
    print(f"Unique cards: {df['hpan'].nunique():,}")
    print(f"Unique customers: {df['pinfl'].nunique():,}")
    print(f"Unique merchants: {df['merchant_name'].nunique():,}")
    print(f"P2P transactions: {df['p2p_flag'].sum():,} ({df['p2p_flag'].mean()*100:.1f}%)")
    print(f"Failed transactions: {(df['respcode'] != '00').sum():,}")
    print(f"Total amount: {df['amount_uzs'].sum():,.2f} UZS")
    print(f"Average amount: {df['amount_uzs'].mean():,.2f} UZS")
    print("\nTop 5 MCC categories:")
    print(df.groupby('mcc')['transaction_code'].count().nlargest(5))
    print("\nTop 5 regions:")
    print(df.groupby('emitent_region')['transaction_code'].count().nlargest(5))


if __name__ == '__main__':
    main()