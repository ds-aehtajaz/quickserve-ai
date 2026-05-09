"""Train the spaCy custom NER model and save it to models/ner/."""
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import spacy
from spacy.training import Example

TRAIN_DATA = [
    ('I want to order 2 pizzas', {'entities': [(17, 18, 'QUANTITY'), (19, 25, 'ITEM')]}),
    ('order me 3 sandwiches', {'entities': [(9, 10, 'QUANTITY'), (11, 21, 'ITEM')]}),
    ('cancel order ORD123', {'entities': [(13, 19, 'ORDER_ID')]}),
    ('track ORD456 for me', {'entities': [(6, 12, 'ORDER_ID')]}),
    ('I need 1 coffee', {'entities': [(7, 8, 'QUANTITY'), (9, 15, 'ITEM')]}),
    ('can I order a burger', {'entities': [(14, 20, 'ITEM')]}),
    ('place order for 5 notebooks', {'entities': [(16, 17, 'QUANTITY'), (18, 27, 'ITEM')]}),
    ('update order ORD789 to 3 burgers', {'entities': [(13, 19, 'ORDER_ID'), (23, 24, 'QUANTITY'), (25, 32, 'ITEM')]}),
    ('where is ORD001', {'entities': [(9, 15, 'ORDER_ID')]}),
    ('get me 4 energy drinks', {'entities': [(7, 8, 'QUANTITY'), (9, 22, 'ITEM')]}),
    ('buy 2 headphones', {'entities': [(4, 5, 'QUANTITY'), (6, 16, 'ITEM')]}),
    ('order ORD222 cancelled', {'entities': [(6, 12, 'ORDER_ID')]}),
    ('I want 6 sodas', {'entities': [(7, 8, 'QUANTITY'), (9, 14, 'ITEM')]}),
    ('please cancel ORD333', {'entities': [(14, 20, 'ORDER_ID')]}),
    ('order 1 laptop bag', {'entities': [(6, 7, 'QUANTITY'), (8, 18, 'ITEM')]}),
    ('status of ORD100', {'entities': [(10, 16, 'ORDER_ID')]}),
    ('modify ORD200 change to sandwich', {'entities': [(7, 13, 'ORDER_ID'), (23, 31, 'ITEM')]}),
    ('2 chicken wraps please', {'entities': [(0, 1, 'QUANTITY'), (2, 15, 'ITEM')]}),
    ('track my ORD500', {'entities': [(9, 15, 'ORDER_ID')]}),
    ('I would like 3 pens', {'entities': [(13, 14, 'QUANTITY'), (15, 19, 'ITEM')]}),
]

nlp = spacy.load('en_core_web_sm')
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner')
else:
    ner = nlp.get_pipe('ner')

for _, ann in TRAIN_DATA:
    for ent in ann['entities']:
        ner.add_label(ent[2])

other_pipes = [p for p in nlp.pipe_names if p != 'ner']
losses_history = []
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.initialize()
    for epoch in range(50):
        random.shuffle(TRAIN_DATA)
        losses = {}
        examples = []
        for text, ann in TRAIN_DATA:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, ann))
        nlp.update(examples, drop=0.3, losses=losses)
        losses_history.append(losses.get('ner', 0))
        if epoch % 10 == 0:
            print(f'Epoch {epoch:2d} | NER loss: {losses.get("ner", 0):.4f}')

print('\nQuick check:')
for text in ['I want 2 pizzas', 'cancel order ORD123', 'track ORD789']:
    doc = nlp(text)
    print(f'  {text!r} -> {[(e.text, e.label_) for e in doc.ents]}')

os.makedirs('models/ner', exist_ok=True)
nlp.to_disk('models/ner')
print('\nNER model saved to models/ner')
