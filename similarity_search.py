from datetime import datetime
from src.database.vector_store import VectorStore
from src.services.syntheizer import Synthesizer

# Initialize the vector store
vec = VectorStore()

# --------------------------------------------------------------
# Shipping question
# --------------------------------------------------------------

relevant_question = "What are your shipping options?"
result = vec.search(relevant_question, limit=3)

response = Synthesizer.generate_response(question=relevant_question, context=result)

print(f"\n{response.answer}")
print("\nThought process:")

for thought in response.thought_process:
    print(f"- {thought}")

print(f"\nContext: {response.enough_context}")
