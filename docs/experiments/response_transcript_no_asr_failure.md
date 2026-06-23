# Response Transcript Without External ASR: Failed Experiment

Date: 2026-06-23

Status: failed and reverted.

## Goal

Remove the generation-time external ASR call by asking Qwen3-Omni to return both:

- `transcript`: current user audio transcription
- `answer`: assistant response to send to TTS

The intended benefit was to keep the backend history text without running a separate ASR model during generation.

## Implementation Tested

Branch:

- `feature/response-transcript-no-asr`

Commits that implemented the failed path:

- `1589204 feat: derive response transcripts from Qwen audio`
- `ddd216a fix: enforce transcript JSON for response calls`

Run:

- `logs/humdial_100_omni_no_asr_transcript_20260623_180323`

The final tested implementation used:

- Qwen3-Omni response calls with a JSON transcript/answer prompt.
- `response_format=json_schema` for the response branch.
- Folded text history inside the current `user` message to avoid Qwen ignoring JSON when normal multi-turn `assistant` history was present.

## Result

The pipeline completed technically:

- noisy: `100/100 ok`
- clean refs: `80/80 ok`
- response events: `162`
- empty response transcripts: `0`

But the score regressed versus the previous Omni run:

| Metric | no-ASR transcript | Previous Omni | Delta |
| --- | ---: | ---: | ---: |
| Interruption Total Score | 72.0 | 78.0 | -6.0 |
| Rejection Total Score | 60.0 | 67.5 | -7.5 |
| Overall Score | 66.0 | 72.75 | -6.75 |
| First Response Delay | 2.221 | 2.202 | +0.019 |
| avg_latency_stop | 0.668 | 1.298 | -0.630 |
| avg_latency_resp | 2.497 | 2.709 | -0.212 |
| Total Delay | 1.795 | 2.070 | -0.275 |

Latency improved, but behavioral quality dropped enough that this path should not be used.

## Main Failure Modes

### 1. Language leakage from Chinese JSON/history prompt

The response prompt and folded history text were Chinese. For English audio, Qwen3-Omni sometimes answered in Chinese despite the original rule saying to answer in the language heard from audio.

Measured on English response events:

- Failed no-ASR run: `13/97` English response events contained Chinese output.
- Previous Omni run: `0/95`.

Examples:

- `repeat_0002_0008`: English task-prioritization prompt, answer became Chinese: `试试四象限法则，分清轻重缓急。`
- `repeat_0005_0039`: English autonomous-weapons prompt, answer became Chinese: `涉及责任归属和人道主义问题。`
- `wait_0010_0007` / `wait_0010_0026`: English stop command, acknowledgement became Chinese, and the evaluation ASR/Judge did not count it as the expected response.

### 2. Folded history changed repeat/resume behavior

Normal multi-turn messages caused Qwen to sometimes ignore the JSON contract, so the implementation folded history into the current `user` content:

```text
用户：...
助手：...
当前用户音频如下。请只输出 JSON。
```

This stabilized transcript extraction, but changed history behavior. Repeat/resume categories became worse because the model no longer saw the same role structure as before.

Examples:

- `repeat_0006_0018_add`: repeated answer was shortened from `Block light, improve sleep quality, enhance privacy.` to `They block light and improve sleep quality.`
- `talk_to_others_0006_0015_add`: third-party resume technically repeated the previous answer, but the previous answer itself drifted from a more specific explanation to a shorter, weaker one.

### 3. Control prompts still occasionally failed

The response branch was structured, but control branches remained free-form.

Example:

- `ask_0007_0002`: shift prompt should return only `yes/no`, but returned a normal assistant answer: `可以利用工作间隙简短交流，或者通过邮件、即时通讯工具了解团队动态。`

That broke the branch logic for that sample.

## Conclusion

The idea worked mechanically: Qwen3-Omni can provide non-empty transcript fields for all response events. However, the required prompt/message wrapping changed model behavior:

- It introduced Chinese-language bias into English samples.
- It weakened history-sensitive behavior, especially repeat/resume.
- It did not remove all free-form control failure modes.

This path is marked failed. The backend should stay on the previous external-ASR generation path unless a future implementation can obtain transcript without changing response prompt language or multi-turn role structure.

## Recommendation For Future Attempts

Do not reintroduce this implementation as-is.

If revisiting the idea, test a design that:

- Uses language-neutral or English-only structural instructions.
- Does not fold history into a Chinese user text block.
- Keeps normal role history if possible.
- Applies structured output separately to control prompts, or hard-validates/retries `judge`, `interrupt`, and `shift`.
- Runs the same 100-sample comparison before accepting the change.
