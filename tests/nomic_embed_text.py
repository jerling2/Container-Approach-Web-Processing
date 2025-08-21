"""
REF: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
"""
import time
import ollama

t_start = time.perf_counter()
response = ollama.embed(
    model='nomic-embed-text',
    input=['The sky is blue because of rayleigh scattering']
)
t_elapsed = time.perf_counter() - t_start

print(f"First 5 values: {response['embeddings'][0][:5]}")
print(f"Elapsed time: {t_elapsed:.2} seconds")