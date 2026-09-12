# Evaluation results

Run on 2026-09-12.
Chat model: `openai/gpt-oss-120b` (provider: groq (sarvam fallback)).
Speech model: `whisper-large-v3-turbo`.
Sarvam was configured but not serving this run: Sarvam is configured but not serving; Groq is answering instead. Rerun after fixing that to get Sarvam's own numbers.

Regenerate with `python eval/run_eval.py`. Numbers move slightly between
runs because the model is not deterministic.

## Memory filter

Does Kivi keep what matters and drop what does not? 18 labelled takes.

Full accuracy (decision and tag correct): **83%**  
Decision only: **89%**  
Wrongly dropped: **0** 

| Decision | Precision | Recall | Cases |
|---|---|---|---|
| save | 0.91 | 0.91 | 11 |
| watch | 0.67 | 0.67 | 3 |
| drop | 1.00 | 1.00 | 4 |

### Failures (3 of 18)

| Take | Wanted | Got | Why it matters |
|---|---|---|---|
| मेरी बहन की शादी दिसंबर में है | save (episode) | save (fact) | Hindi: my sister's wedding is in December |
| I think I might be coming down with something | watch (episode) | save (episode) | hedged, uncertain permanence |
| Remind me to call the plumber | save (episode) | watch (preference) | a task the user wants kept |

## Retrieval

Does the right memory come back for a question? 14 targeted queries
plus 3 that should find nothing.

hit@1: **79%**  
hit@3: **79%**  
hit@5: **86%**  
Mean top score: 0.477

| Query | Wanted | Result |
|---|---|---|
| Where do I work? | Zoho | not retrieved **FAIL** |
| What is my blood group? | O positive | rank 1 |
| Who is my manager? | Rahul | rank 1 |
| What colour is Pixel? | black and white | rank 1 |
| When is my dentist appointment? | Tuesday | rank 1 |
| When is my sister getting married? | December | rank 1 |
| What coffee do I drink? | filter coffee | rank 1 |
| How do I sign off my emails? | Warm regards | rank 1 |
| Am I allergic to anything? | peanut | rank 4 **FAIL** |
| Where do I live? | Adyar | rank 1 |
| What time of day do I work best? | morning | rank 1 |
| எனக்கு எந்த ரத்த வகை? | O positive | rank 1 |
| मेरा ऑफिस कहाँ है? | Zoho | not retrieved **FAIL** |
| How do I get to work? | cycle | rank 1 |
| What is the capital of Mongolia? | nothing | no target, handled by the citation gate |
| Do I have a dog? | nothing | no target, handled by the citation gate |
| What is my father's name? | nothing | no target, handled by the citation gate |

## Speech to text

Word error rate on dictation. English cases are synthesised locally so
the suite runs on any machine. Tamil, Hindi and code-mixed cases are
listed but not scored: a synthetic voice reading them would not tell
you anything about how a person actually speaks.

Mean WER on scored cases: **4.6%** (8 scored, 3 untested)

| Case | WER | Heard |
|---|---|---|
| a01 (en) | 8.3% | I have a dentist appointment next Tuesday at 3 in the af |
| a02 (en) | 10.0% | My manager is Rahul and my skip level is Davya. |
| a03 (en) | 0.0% | I work at Zoho in Chennai as a data analyst. |
| a04 (en) | 10.0% | My sister Kavya is getting married in Combattore in Dece |
| a05 (en) | 0.0% | I always order filter coffee when I am working late. |
| a06 (en) | 8.3% | I cycle to work most mornings and it takes about 40 minu |
| a07 (en) | 0.0% | I am allergic to peanuts quite badly. |
| a08 (en) | 0.0% | Book the usual table at Ananda for Friday evening. |
| a09 (ta) | not scored | untested, needs a human recording |
| a10 (hi) | not scored | untested, needs a human recording |
| a11 (ta-en) | not scored | untested, needs a human recording |
