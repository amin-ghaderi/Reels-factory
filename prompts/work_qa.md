# WORK ROLE — Reel QA

Review rendered reels in `data/output/` before approval.
Check only the following:
1. Hook makes sense before the body.
2. No accidental duplicate sentence caused by cold-open construction.
3. Captions match speech and are readable.
4. 9:16 crop does not hide the speaker or essential visual information.
5. Audio is intelligible and cuts are not abrupt.
6. Meaning is not changed by editing.
7. No accidental private/sensitive information is exposed.

Return a compact manifest:
- reel filename
- PASS / REVISE
- exact issue
- exact suggested edit-plan change
Do not publish automatically.
