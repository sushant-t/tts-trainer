from utils.smart_splitting import process_split
from utils.transcribe import transcribe_all
import os

def generate_data(path):
    print("processing file {0}".format(path))
    
    print("splitting file...")
    process_split(path)
    
    base_name = os.path.splitext(os.path.basename(path))[0]
    
    print("transcribing file segments...")
    transcribe_all(base_name)
    