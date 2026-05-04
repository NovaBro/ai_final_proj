# Project Overview / Changes
This is just an overview of major code changes made. 
Many small changes as requested in the notebook files are made, but those code are not copied here for the sake of brevity. 

Google Drive For Videos: https://drive.google.com/drive/folders/1qx0KjJWLEjmWXRGFIlVKGupMlqXOxnku?usp=sharing

## Notebook-1 Download Integration
No Changes
## Notebook-2 Transcription Integration
No Changes

## Notebook-3 Translation Integration
Followed instructions in `translation_integration.ipynb` in notebooks folder.<br>
What was modified: `foreign_whispers/reranking.py`.<br>
Task1: Tried to use summarization, did not yield better results, probably due to the fact that the text itself is too short to be summarized even further. 
We simply went to use a different translation model. Did not yield much better shorter translations.
I have added a comment `# Task 2 in Notebook 5` in the `reranking.py` for when further modifcations were needed for Task 2 in Notebook 5, re-generating translations
```python
# foreign_whispers/reranking.py
# ...
    canidate_list = []

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Device: {device}")

    # Summarization
    # from transformers import pipeline
    # summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    # print(f"Original Text: {source_text}")
    # summarized_text = summarizer(source_text, max_length=45, min_length=0, do_sample=False)
    # source_text = summarized_text['summary_text']
    # print(f"Summarized Text: {source_text}")

    # Translation
    from transformers import MarianMTModel, MarianTokenizer
    model_name = "Helsinki-NLP/opus-mt-tc-big-en-es"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    inputs = tokenizer(source_text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # translated = model.generate(**inputs)

    # Generate multiple candidates
    translated = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        num_return_sequences=3  
    )

    # Decode the translations
    for t in translated:
        translation_result = tokenizer.decode(t, skip_special_tokens=True)
        # print(f"translation_result: {translation_result}")
        canidate =  TranslationCandidate(translation_result, len(translation_result), 'None')
        canidate_list.append(canidate)

    # Task 2 in Notebook 5
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model_semantic = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    def semantic_distance(text1, text2, model=model_semantic):
        emb1 = model.encode([text1])[0]
        emb2 = model.encode([text2])[0]
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return 1 - cosine_sim
    lamb = 0.1
    from foreign_whispers.alignment import _estimate_duration
    smallest_candidate_score = float('inf')
    smallest_candidate = None
    for candidate in canidate_list:
        candidate_duration = _estimate_duration(candidate.text)
        candidate_dist = semantic_distance(baseline_es, candidate.text, model=model_semantic)
        candidate_score = (candidate_duration - target_duration_s) ** 2 + lamb * candidate_dist
        print(f"Candidate: {candidate.text}, Score: {candidate_score}")
        if candidate_score < smallest_candidate_score:
            smallest_candidate_score = candidate_score
            smallest_candidate = candidate

    return [smallest_candidate]
```


## Notebook-4 Diarization
Followed instructions in `diarization_integration.ipynb` in notebooks folder.<br>
What was modified: `foreign_whispers/diarization.py`.<br>
Task 1: Passed the tests, shown in notebook.
```python
def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is
    empty, all segments default to ``SPEAKER_00``.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        Original list is not mutated.
    """
    # ---- YOUR CODE HERE ----
    # raise NotImplementedError("Implement this function")
    segments_with_speaker = []
    if diarization:
        for seg in segments:
            seg_copy = copy.copy(seg)
            seg_end = seg_copy['end']
            seg_start = seg_copy['start']

            largest_overlap = 0
            largest_speaker = None
            for dia in diarization:
                overlap = max(0, min(seg_end, dia['end_s']) - max(seg_start, dia['start_s']))
                if overlap > largest_overlap:
                    largest_speaker = dia['speaker']
            seg_copy['speaker'] = largest_speaker
            segments_with_speaker.append(seg_copy)

    else:
        for seg in segments:
            seg_copy = copy.copy(seg)
            seg_copy['speaker'] = 'SPEAKER_00'
            segments_with_speaker.append(seg_copy)

    return segments_with_speaker
    # ---- END YOUR CODE ----
```

Task 2: Simply followed what was instructed in the notebookfile.
Modified `api/src/routers/diarize.py`, `api/src/schemas/diarize.py`, `api/src/main.py`, `api/src/core/config.py`

Task 3: Added the code near line 96 of `api/src/routers/diarize.py`

Task 4: Simply followed what was instructed in the notebookfile.

Task 5: Wasn't entirely sure on implementation details since "**This is open-ended** — use the existing `text_file_to_speech` function as your starting point.".
I made modifications to `tts_service.py` and `tts_engine.py`. In order to assign difference voices to different segments, we need to modify the client to use a different wave file. This can only be done in the `tts_engine.py` file, or pulling code from the engine file to ther service file. If this was not the implementaiton intention, please let me know. I would appreciate knowing what was inteded. <br>

```python 
# tts_service.py
# Strategy 3: Assigning voice wav files for each speaker. Reasoning is located in the notebook file.
class TTSService:
    """Thin wrapper around the TTS pipeline.

    Accepts *ui_dir* and a pre-loaded *tts_engine* via constructor injection.
    """

    def __init__(self, ui_dir: Path, tts_engine: Any) -> None:
        self.ui_dir = ui_dir
        self.tts_engine = tts_engine

    def text_file_to_speech(self, source_path: str, output_path: str, *, alignment: bool | None = None, speaker_wav: str | None = None) -> None:
        """Generate time-aligned TTS audio from a translated JSON transcript."""
        reference_voice = {
            "defalt" : "default.wav",
            "SPEAKER_00" : "SPEAKER_00.wav",
            "SPEAKER_01" : "SPEAKER_01.wav",
            "SPEAKER_02" : "SPEAKER_02.wav",
        }

        tts_text_file_to_speech(source_path, output_path, self.tts_engine, alignment=alignment, speaker_wav=speaker_wav, reference_voice=reference_voice)
```

In the engine file, I changed the `text_file_to_speech` function and corresponding `_do_synth` and `_synthesize_raw` function. Here, we pass the `speaker_wav` keyword param to `tts_engine.tts_to_file` so the Chatter box client uses a different speaker for each segment.
```python 
# tts_engine.py
with tempfile.TemporaryDirectory() as synth_dir:
    def _do_synth(idx: int, text: str, speaker_wav:str) -> tuple[int, bytes | None]:
        wav_path = str(pathlib.Path(synth_dir) / f"seg_{idx}.wav")
        return idx, _synthesize_raw(engine, text, wav_path, speaker_wav)

    with ThreadPoolExecutor(max_workers=_TTS_WORKERS) as pool:
        futures = {
            pool.submit(_do_synth, m["index"], m["text"], m['speaker_wav']): m["index"]
            for m in seg_metas
        }
        for fut in as_completed(futures):
            idx, raw_bytes = fut.result()
            raw_wav_map[idx] = raw_bytes
# ...

def _synthesize_raw(tts_engine, text: str, wav_path: str, speaker_wav:str) -> bytes | None:
    """GPU-bound: call TTS engine and return raw WAV bytes, or None on failure."""
    if not text or not text.strip():
        return None
    try:
        tts_engine.tts_to_file(text=text, file_path=wav_path, speaker_wav=speaker_wav)
        return pathlib.Path(wav_path).read_bytes()
    except Exception as exc:
        print(f"[tts] TTS failed for segment ({exc}), using silence")
        return None
```


## Notebook-5 Alignment Integration

Some of my code exprimentaiton is in the `notebooks/alignment_integration/alignment_integration.ipynb` notebook.

Task 1: I tried using a fitted linear regression model to try to predict the time for a text segment. I test various basic features and downloaded an external library to help syllablize the sentece. However, despite all these more advance features, there was not a significant improvement.

```python
# foreign_whispers/alignment.py
def _estimate_duration(text: str) -> float:
    """Estimate TTS duration in seconds using a syllable-rate heuristic."""
    # NOTE: BASELINE: baseline achieves higher accuracy than the model below
    # return _count_syllables(text) / _SYLLABLE_RATE

    from sklearn.linear_model import LinearRegression
    import numpy as np
    import re
    import silabeador

    model = LinearRegression()
    model.coef_ = np.array([0.07759564, -0.02432792, -0.02116867, 0.00372126, 0.01143961])
    model.intercept_ = 0.5680703852016769

    def estimate_syllables(word: str) -> int:
        w = re.sub(r'[^a-z]', '', word.lower())
        if not w:
            return 0
        groups = re.findall(r'[aeiouy]+', w)
        s = len(groups)
        if w.endswith('e') and s > 1:
            s -= 1
        return max(1, s)

    def embed_text(text):
        f1 = len(text)
        # f2 = seg['speed_factor'] # This feature not available in alignment.py
        f2 = estimate_syllables(text)
        word_list = text.split()
        f3 = len(word_list)

        # words_len = [len(x) for x in word_list]
        # f4 = np.median(words_len)
        # f5 = np.std(words_len)
        # f6 = np.mean(words_len)
        if text[0] == 'y':
            f6 = len(silabeador.syllabify(text[1:]))
        else:
            f6 = len(silabeador.syllabify(text))
        # f7 = np.max(words_len)
        f7 = _count_syllables(text)
        # f8 = np.min(words_len)

        # vector_embed = np.array([f1, f2, f3, f4, f5, f6, f7, f8])
        # vector_embed = np.array([f1, f2, f3, f4, f5, f6])
        # vector_embed = np.array([f1, f2, f3, f6])
        vector_embed = np.array([f1, f2, f3, f6, f7])
        return vector_embed


    return model.predict([embed_text(text)])[0]
```

Task 2: The results of the reranking is dependent on lamb, the hyperparameter to judge how much to preserve semantics. Further more, the temperature of the model generation also has a major impact on the translations canidates given. The result, it did manage to reduce some of the `request_shorter` segments, but not by a major margin.
```python
# foreign_whispers/reranking.py
# ...
    # Generate multiple candidates
    translated = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        num_return_sequences=3  
    )
# ...
    # Task 2 in Notebook 5
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model_semantic = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    def semantic_distance(text1, text2, model=model_semantic):
        emb1 = model.encode([text1])[0]
        emb2 = model.encode([text2])[0]
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return 1 - cosine_sim
    lamb = 0.1
    from foreign_whispers.alignment import _estimate_duration
    smallest_candidate_score = float('inf')
    smallest_candidate = None
    for candidate in canidate_list:
        candidate_duration = _estimate_duration(candidate.text)
        candidate_dist = semantic_distance(baseline_es, candidate.text, model=model_semantic)
        candidate_score = (candidate_duration - target_duration_s) ** 2 + lamb * candidate_dist
        print(f"Candidate: {candidate.text}, Score: {candidate_score}")
        if candidate_score < smallest_candidate_score:
            smallest_candidate_score = candidate_score
            smallest_candidate = candidate

    return [smallest_candidate]
```

Task 3:

Unsure how to do this implementation fully, inparticular hard to apply solutions list to this problem.

Task 4:

Unsure how to do this implementation, particularly the semantic search.
Tried displaying metrics to timing accuracy instead. Within `evaluation.py`

## Notebook-6 TTS Integration

Task 1:

No Changes

Task 2:

Followed instructions, step by step, logic sequence. 

```python
def resolve_speaker_wav(
    speakers_dir: Path,
    target_language: str,
    speaker_id: str | None = None,
) -> str:
    """
    # 1. speakers/{lang}/{speaker_id}.wav
    if speaker_id:
        specific_path = Path(target_language) / f"{speaker_id}.wav"
        if (speakers_dir / specific_path).is_file():
            return str(specific_path)

    # 2. speakers/{lang}/default.wav
    lang_default = Path(target_language) / "default.wav"
    if (speakers_dir / lang_default).is_file():
        return str(lang_default)

    # 3. speakers/default.wav
    return "default.wav"
```

Task 3 / 4:
I mostly followed what was perscribed in the notebook instructions, but I think due to earlier design choices, my current implementation had to diverge somewhat. I think Task 4 is meant to override implementations done in Task 3, which is speader_wav is not used in the tts_endpoint. But I still included some code related for sake of completeness. The method for assigning voice wav files is given to voice_mask in Task 4, instead of speaker_wav as told in Task 3. The code now reflects the needs of task 4. Worth noting, because of this, the notebooke task 3.4 doesn't work, since the api route is changed for task 4 use. 

In the notebook instruction:
"""
Currently `text_file_to_speech` synthesizes all segments with one `speaker_wav`. You need to modify it so that each segment can use a different speaker_wav based on its `speaker` field.

**Approach:** Pass `voice_map` as a dict to `text_file_to_speech`. Inside the function, for each segment, look up `voice_map[segment["speaker"]]` and pass it as `speaker_wav` to `tts_to_file()`.
"""

Since there were a few Tasks before that were "open-ended" (diarization notebook) `voice_map[segment["speaker"]]` is actually done within `tts_engine.py`. This changes the api.

```python
# tts_service.py
    def text_file_to_speech(self, source_path: str, output_path: str, *, alignment: bool | None = None, voice_map: dict = None) -> None:
        """Generate time-aligned TTS audio from a translated JSON transcript."""
        # Depricated old method, used for Task 5 in Notebook 4
        # reference_voice = {
        #     "defalt" : "default.wav",
        #     "SPEAKER_00" : "SPEAKER_00.wav",
        #     "SPEAKER_01" : "SPEAKER_01.wav",
        #     "SPEAKER_02" : "SPEAKER_02.wav",
        # }

        tts_text_file_to_speech(source_path, output_path, self.tts_engine, alignment=alignment, reference_voice=voice_map)
```

```python
# tts.py
@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    speaker_wav: str = Query(None, description="Reference voice WAV path (e.g. 'es/default.wav')"),
):
# ...
    # Load translated transcript to get speaker labels
    trans_path = settings.translations_dir / f"{title}.json"
    translated = json.loads(trans_path.read_text())
    segments = translated.get("segments", [])

    # Build speaker → voice mapping
    # unique_speakers = sorted(
    #     # {seg.get("speaker") or "SPEAKER_00" for seg in segments}
    #     {seg.get("speaker") or "SPEAKER_00" for seg in segments}
    # )
    unique_speakers = sorted(set(seg.get("speaker", "SPEAKER_00") for seg in segments))
    voice_map = {
        spk: resolve_speaker_wav(settings.speakers_dir, "es", spk)
        for spk in unique_speakers
    }

    await _run_in_threadpool(
        None, svc.text_file_to_speech, source_path, str(audio_dir), alignment=alignment, voice_map=voice_map
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
    }
```

## Notebook-7 Stitch Integration
No Changes

## OTHER NOTES:

NOTE: The other speaker voice wave files are copies.

NOTE: The 4th video (IiBKsv-D64M) in the yaml file was unable to be downloaded due to this error. Other videos worked fine. No changes were made in the download process.
```
  File "/app/api/src/services/download_service.py", line 15, in dv_download_video
    return _dl.download_video(url, destination, filename)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/api/src/services/download_engine.py", line 66, in download_video
    ydl.download([url])
  File "/app/.venv/lib/python3.11/site-packages/yt_dlp/YoutubeDL.py", line 3670, in download
    self.__download_wrapper(self.extract_info)(
  File "/app/.venv/lib/python3.11/site-packages/yt_dlp/YoutubeDL.py", line 3643, in wrapper
    res = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.11/site-packages/yt_dlp/YoutubeDL.py", line 1687, in extract_info
    return self.__extract_info(url, self.get_info_extractor(key), download, extra_info, process)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/.venv/lib/python3.11/site-packages/yt_dlp/YoutubeDL.py", line 1716, in wrapper
    self.report_error(str(e), e.format_traceback())
  File "/app/.venv/lib/python3.11/site-packages/yt_dlp/YoutubeDL.py", line 1154, in report_error
    self.trouble(f'{self._format_err("ERROR:", self.Styles.ERROR)} {message}', *args, **kwargs)
  File "/app/.venv/lib/python3.11/site-packages/yt_dlp/YoutubeDL.py", line 1093, in trouble
    raise DownloadError(message, exc_info)
yt_dlp.utils.DownloadError: ERROR: [youtube] IiBKsv-D64M: Requested format is not available. Use --list-formats for a list of available formats
INFO:     127.0.0.1:58230 - "GET /api/video/IiBKsv-D64M/original HTTP/1.1" 404 Not Found
INFO:     127.0.0.1:58238 - "GET /api/captions/IiBKsv-D64M/original HTTP/1.1" 404 Not Found
INFO:     127.0.0.1:52850 - "GET /api/videos HTTP/1.1" 200 OK
```