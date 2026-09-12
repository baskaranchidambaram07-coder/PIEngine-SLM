# PII Guard — what is checked and how a refusal works

## What the guard looks for
The PII Guard reviews every attached file and every message before the assistant answers. It looks for:
- Secrets: passwords, PINs, API keys (AWS, GitHub, Google, Stripe, OpenAI, Anthropic, Hugging Face, Slack, SendGrid), bearer tokens, JWTs, private key blocks, and labelled values such as `api_key = ...`.
- Identity data: payment card numbers, Emirates ID (UAE) numbers, driving licence numbers (India, UK, or any licence given by name), US Social Security numbers, Aadhaar numbers, PAN numbers, IBANs, passport numbers, national or citizen ID cards, voter IDs, health or insurance card numbers, and dates of birth.
- Contact data: email addresses and phone numbers.
- With the model judge: home addresses, government ID numbers, bank account numbers and health or salary details tied to a person.

## What happens when something is found
The request is refused before any answer is generated. The refusal lists each finding by category and shows only the first three and last two characters of the value; everything else is masked. The message ends with "I am not ready to proceed further" and asks the user to remove or redact the information and try again.

## Supported attachments
One file per message: jpg, jpeg, pdf, txt or md, up to 5 MB. Scanned PDFs and photos are read with OCR, and images are also read by an on-device vision model.

## What is not treated as sensitive
Company or shop addresses, invoice and order numbers, prices, totals, ordinary dates, product names and job titles are business data and do not trigger a refusal. Personal names alone do not trigger a refusal unless the deployment turns that policy on.
