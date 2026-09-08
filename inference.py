"""Run NV-Reason-CT inference on one NIfTI CT volume."""

import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "nvidia/NV-Reason-CT"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ct_path", help="Path to a .nii or .nii.gz CT volume")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Local model directory or Hugging Face model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--region",
        choices=("chest", "abdomen"),
        default="chest",
        help="Anatomy-aware crop to use (default: chest)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Question or instruction (default: a structured report for the selected region)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen3.5 thinking mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NV-Reason-CT inference requires a CUDA-capable GPU")

    default_prompts = {
        "chest": "write a structured chest CT report",
        "abdomen": "write a structured abdominal CT report",
    }
    prompt_text = args.prompt or default_prompts[args.region]

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().to("cuda")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not args.disable_thinking,
    )
    inputs = processor(
        text=text,
        images3d=[args.ct_path],
        anatomy_region=args.region,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    new_tokens = generated_ids[:, inputs.input_ids.shape[1] :]
    response = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(response)


if __name__ == "__main__":
    main()
