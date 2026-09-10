"""Templated stand-in corpora for training data.

Documents here are `.md.tmpl` files with `{{placeholder}}` slots, rendered with
a fresh entity cast per example (see entities.py). Never put real customer
content in this directory — the whole point of the surrogates is that nothing
in a customer's knowledge base ever reaches the weights.
"""
