from contextual_privacy_llm import PrivacyAnalyzer, run_single_query

analyzer = PrivacyAnalyzer(
    model="llama3.2:3b-instruct-fp16",
    prompt_template="llama",
    experiment="dynamic"
)
t = """Hey everyone!! I’m applying for the B.Sc. Computer Science (English) program for Winter 2026/27, and I was just curious how many others are applying this cycle too in December. If you’re applying, feel free to drop a comment... would be nice to see who all are in the same boat and where everyone’s applying from. edit: I'm applying from India Which subjects should I revise before the studying term"""
result = run_single_query(
    query_text=t,
    query_id="001",
    model="llama3.2:3b-instruct-fp16",
    prompt_template="llama",
    experiment="dynamic"
)

print(result['reformulated_text'])
# → "What autism support exists for parents in Paris?"

print(result)

# → {
# →   "query_id": "001",
# →   "original_text": "My child has autism and I’m in Paris. What support exists for moms like me?",
# →   "intent": "support_seeking",
# →   "task": "resource_lookup",
# →   "related_context": ["autism", "Paris"],
# →   "not_related_context": ["moms like me", "my child"],
# →   "reformulated_text": "What autism support exists in Paris?"
# → }