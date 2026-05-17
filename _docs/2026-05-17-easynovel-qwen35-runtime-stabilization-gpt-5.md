# Goal

Stabilize the Q8_0 Huihui Qwen3.5 roleplay GGUF when used from EasyNovelAssistant through the KoboldCpp API, and separate runtime/template issues from the need for further QLoRA.

# Files touched

- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\setup\res\default_llm_sequence.json`
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\kobold_cpp.py`
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\generator.py`
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\tests\test_backend_selection.py`

# Key decisions

- Treat the GGUF as structurally valid and focus first on EasyNovelAssistant/KoboldCpp runtime behavior.
- Add a `QwenChatML` sequence for Huihui/Qwen/Qwen3.5 GGUF names with a Japanese system prompt, ChatML user/assistant wrapping, Qwen stop sequences, conservative sampler overrides, and `max_length=768`.
- Strip Qwen thinking blocks and a short leading meta phrase from generated text before it reaches the EasyNovelAssistant output area.
- Avoid duplicating the last input line when the model prompt is auto-wrapped as ChatML.
- Recover a remembered `[直接選択] ...` GGUF after app restart when the copied GGUF exists under the local `KoboldCpp` directory; otherwise the app can silently fall back to a different model and lose the Qwen sequence.

# Verification

- `py -3 -m pytest EasyNovelAssistant\tests\test_backend_selection.py -q` -> `13 passed`.
- `py -3 -m py_compile EasyNovelAssistant\src\kobold_cpp.py EasyNovelAssistant\src\generator.py EasyNovelAssistant\tests\test_backend_selection.py` -> passed.
- Current EasyNovelAssistant config smoke confirmed:
  - selected model: `[直接選択] huihui-qwen35-4b-roleplay-unsloth-qlora-claude35-15k-ms2048-s110-q8_0`
  - file: `huihui-qwen35-4b-roleplay-unsloth-qlora-claude35-15k-ms2048-s110-q8_0.gguf`
  - `prompt_first_line=<|im_start|>system`
  - `max_context_length=4096`
  - `max_length=768`
  - `temperature=0.55`, `top_p=0.9`, `top_k=40`, `rep_pen=1.08`, `min_p=0.02`
- Live KoboldCpp API smoke with the EasyNovelAssistant `KoboldCpp.generate()` path returned coherent Japanese continuation text without raw `<think>` output.

# Next-session notes

- If output quality is still unsatisfactory after restarting EasyNovelAssistant and KoboldCpp with this patch, continue QLoRA from a cleaner training plan rather than assuming GGUF conversion corruption.
- The current adapter is only the short `s110` run. A stronger follow-up should train at stable `max_seq_length=2048`, use the filtered 15.3k-formatted set plus carefully filtered additional examples, track validation loss, and evaluate with the actual EasyNovelAssistant prompt path before exporting another Q8_0 GGUF.
