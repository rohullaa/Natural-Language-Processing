from collections import Counter
from dotenv import load_dotenv

from neurotic.nlp.parts_of_speech.preprocessing import DataLoader
from neurotic.nlp.parts_of_speech.training import TheTrainer

load_dotenv("posts/nlp/.env")
loader = DataLoader()
trainer = TheTrainer(loader.processed_training)
