"""Run once to initialise the database and load products + FAQ KB."""
import csv
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quickserve.db.models import init_db, SessionLocal, Product, FaqKb

init_db()
db = SessionLocal()

# Products
with open('data/products_seed.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not db.query(Product).filter_by(id=int(row['id'])).first():
            db.add(Product(
                id=int(row['id']),
                name=row['name'],
                category=row['category'],
                price=float(row['price']),
                stock=int(row['stock']),
                description=row['description'],
            ))

# FAQ KB
with open('data/faq_kb.json', encoding='utf-8') as f:
    faqs = json.load(f)
if db.query(FaqKb).count() == 0:
    for item in faqs:
        db.add(FaqKb(question=item['question'], answer=item['answer']))

db.commit()
db.close()
print('Database seeded successfully.')
