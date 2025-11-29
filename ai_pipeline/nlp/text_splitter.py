from ai_pipeline.config.settings import MAX_CHUNK_SIZE

def split_text(text, chunk_size=MAX_CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
