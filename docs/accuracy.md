# Accuracy and performance

How the numbers in the README and on the [model card](https://huggingface.co/maolon/WeMM-Embedding-2B-CoreML-ANE) were measured.

## Method

- **Reference.** The original [tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B) in fp32 (PyTorch, CPU), with its own processor and chat template, exactly as its model card shows.
- **Candidate.** `embed-ane serve` on an M1 Pro running macOS 27, with the release model bundle. Requests go through the HTTP endpoint, so the numbers include tokenization, image and video preprocessing, the Core ML cascade and JSON transport.
- **Metric.** Cosine similarity between the two 2,048-dimensional embeddings of the same input.
- **Prompt check.** For every case, the number of prompt tokens reported by the server equals the reference processor's sequence length. Text, image and video prompts are therefore built token-for-token like upstream.

## Cases and results

| Input | Prompt tokens | Cosine to fp32 |
| --- | --- | --- |
| "hello world" | 7 | 0.990 |
| "How is mapo tofu prepared?" | 12 | 0.970 |
| "The Apple Neural Engine runs transformer encoders efficiently." | 15 | 0.985 |
| A Chinese sentence about multimodal embeddings | 20 | 0.977 |
| "Silent fanless embedding service on Apple Silicon." | 15 | 0.966 |
| A Chinese sentence about retrieval-augmented generation | 14 | 0.977 |
| Image (672×448 synthetic shapes) + caption | 306 | 0.949 |
| Video: 57 s, 360×640, 30 fps clip; 8 sampled frames at 448×256 | 488 | 0.920 |

## Where the gap comes from

- **fp16 decoder.** Almost all of the difference is fp16 rounding in the converted decoder. It grows with sequence length: text prompts are short, and the image and video prompts are 306 and 488 tokens. The vision tower contributes little: per-token cosine against fp32 has a median of 0.998–0.9995.
- **Video decoding.** AVFoundation and the reference's ffmpeg decoder produce slightly different pixels (mean 2 levels out of 255). On fp32 alone this costs 0.006 cosine.
- **Not a source of error.** The image resize policy (`smart`, the default) reproduces the reference image size exactly, and the per-frame-pair vision tower reproduces the reference video features (fp32 cosine ≥ 0.9999996).

## Video sampling

The reference processor samples video at 2 fps (up to 768 frames) at high resolution. The decoder here holds 512 tokens, so Embed ANE samples 4–8 frames uniformly over the clip and lowers the resolution to fit. The numbers above compare both sides on the same 8 frames. Embeddings of long or detailed videos will differ further from what the original model produces with its default sampling.
