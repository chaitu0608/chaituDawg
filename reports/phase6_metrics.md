# Phase 6 Metrics Report

Generated at (UTC): 2026-04-22T20:03:36+00:00

| Metric | Value | Target | Passed |
|---|---:|---:|:---:|
| intent_classification_accuracy | 100.00% | 95.00% | yes |
| rag_factual_accuracy | 100.00% | 100.00% | yes |
| tool_call_precision | 100.00% | 100.00% | yes |
| lead_slot_completion | 100.00% | 100.00% | yes |
| memory_retention | 100.00% | 100.00% | yes |

Overall Passed: yes

## Details
- intent_classification_accuracy: {'samples': 19, 'correct': 19, 'mismatches': []}
- rag_factual_accuracy: {'samples': 6, 'exact_matches': 6, 'mismatches': []}
- tool_call_precision: {'true_positives': 2, 'false_positives': 0, 'false_negatives': 0, 'mismatches': []}
- lead_slot_completion: {'sessions': 3, 'successful': 3, 'failures': []}
- memory_retention: {'turn_count_at_least_six': True, 'twelve_messages_stored': True, 'lead_name_persisted': True, 'lead_email_persisted': True, 'lead_platform_persisted': True, 'lead_captured': True, 'restored_recent_context_has_twelve_messages': True}
