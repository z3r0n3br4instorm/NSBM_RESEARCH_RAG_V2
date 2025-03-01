# Issues

DATE - ID - Description
25/2/2025 - 1 - maxtoken length of the model is limited to 512 tokens
1//3/2025 - 2 - discovered that the 512 token length issue arose 
                because we were using the `tiny-dolphin` model which is relatively a very small model.

# Solutions

DATE - ID - Description
25/2/2025 - 1 - tokenized output was checked and sliced to 512
1//3/2025 - 2 - Switched to the relatively larger counterpart of it, that is based on llama-3.2, which is 
                `cognitivecomputations/Dolphin3.0-Llama3.2-3B`. And added the truncation system as well just in case

# TODO
- Implementing memory chain
