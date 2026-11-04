# ACE-Step 1.5 launcher for Quadro T2000 Max-Q 4 GB + 32 GB RAM (Gradio UI, loopback only).
# DiT turbo 2B: CUDA, fp32 compute (ACESTEP_DTYPE), INT8 weight-only, full CPU offload.
# 5Hz LM 1.7B: CPU fp32, pt backend (enables thinking / CoT / sample mode).
# Do not run while ComfyUI or GPU TTS is using the GPU.
param(
    [switch]$NoLM,
    [int]$Port = 7860
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:ACESTEP_LM_DEVICE = "cpu"
# fp16 on T2000 (TU117) produces NaN latents; fp32 + INT8 weights fits in 4 GB.
$env:ACESTEP_DTYPE = "float32"
$env:ACESTEP_COMPILE_MODEL = "false"
$env:PYTHONUTF8 = "1"

$initLlm = if ($NoLM) { "false" } else { "true" }

uv run acestep `
    --server-name 127.0.0.1 `
    --port $Port `
    --init_service true `
    --config_path acestep-v15-turbo `
    --device cuda `
    --offload_to_cpu true `
    --offload_dit_to_cpu true `
    --quantization int8_weight_only `
    --init_llm $initLlm `
    --lm_model_path acestep-5Hz-lm-1.7B `
    --backend pt
