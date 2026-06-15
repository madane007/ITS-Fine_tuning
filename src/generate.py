# src/generate.py

import torch

def generate(model, tokenizer, system_prompt, prompt, max_new_tokens=512):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize              = True,
        add_generation_prompt = True,
        return_tensors        = "pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids      = inputs,
            max_new_tokens = max_new_tokens,
            temperature    = 0.7,
            top_p          = 0.9,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)