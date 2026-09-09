import logging
import re

try:
    from .vlm_labels import CTLabels
except ImportError:
    from vlm_labels import CTLabels

logger = logging.getLogger(__name__)


REGION_LABELS = {
    "chest": frozenset(label.lower() for label in CTLabels.get_chest_list()),
    "abdomen": frozenset(label.lower() for label in CTLabels.get_abdomen_list()),
}
REGION_NO_FINDING_LABEL = {
    "chest": CTLabels.No_Chest_Finding.lower(),
    "abdomen": CTLabels.No_Abdominal_Finding.lower(),
}


def _parse_label_set(text: str) -> set[str]:
    """Parse comma-separated labels without silently dropping unknown labels."""
    return {label.strip().lower() for label in text.strip().split(",") if label.strip()}


def _normalize_anatomy_regions(
    anatomy_region,
    expected_length: int,
    reward_name: str,
) -> list[str]:
    if isinstance(anatomy_region, str) or anatomy_region is None:
        regions = [anatomy_region] * expected_length
    else:
        regions = list(anatomy_region)

    if len(regions) != expected_length:
        raise ValueError(
            f"{reward_name}: anatomy_region length ({len(regions)}) must match "
            f"number of completions ({expected_length})"
        )

    normalized_regions = []
    for index, region in enumerate(regions):
        normalized_region = str(region).strip().lower() if region is not None else ""
        if normalized_region not in REGION_LABELS:
            raise ValueError(
                f"{reward_name} requires anatomy_region 'chest' or 'abdomen'; "
                f"got {normalized_region!r} at index {index}"
            )
        normalized_regions.append(normalized_region)
    return normalized_regions


def _parse_and_validate_gold_labels(
    solution: str,
    anatomy_region: str,
    index: int,
    reward_name: str,
) -> set[str]:
    labels = _parse_label_set(solution)
    if not labels:
        raise ValueError(
            f"{reward_name}: solution must contain an explicit {anatomy_region} "
            f"label at index {index}"
        )

    invalid_labels = labels - REGION_LABELS[anatomy_region]
    if invalid_labels:
        raise ValueError(
            f"{reward_name}: solution contains labels outside the "
            f"{anatomy_region} schema at index {index}: {sorted(invalid_labels)}"
        )

    no_finding_label = REGION_NO_FINDING_LABEL[anatomy_region]
    if no_finding_label in labels and len(labels) != 1:
        raise ValueError(
            f"{reward_name}: solution combines {no_finding_label!r} with "
            f"other labels at index {index}"
        )
    return labels


def _prediction_is_contradictory(labels: set[str], anatomy_region: str) -> bool:
    no_finding_label = REGION_NO_FINDING_LABEL[anatomy_region]
    return no_finding_label in labels and len(labels) != 1


def _extract_answer_text(content: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    return match.group(1) if match else None


def structured_report_format_reward(completions, anatomy_region=None, **kwargs) -> list[float]:
    """Reward the current region-specific structured-report scaffold.

    The reward is adaptive: each completion is checked against either the
    chest or abdomen schema selected by that sample's ``anatomy_region``.
    ``TECHNIQUE`` is optional, and legacy ``INDICATION`` / ``COMPARISON`` /
    ``UPPER ABDOMEN`` headings are not rewarded.
    """
    region_headers = {
        "chest": (
            "Medical devices",
            "CHEST",
            "Lungs and airways",
            "Pleura",
            "Mediastinum and hila",
            "Heart and pericardium",
            "Great vessels",
            "Esophagus",
            "Chest wall and breast",
            "Bones",
        ),
        "abdomen": (
            "Medical devices",
            "ABDOMEN",
            "Liver and biliary system",
            "Spleen",
            "Pancreas",
            "Adrenal glands",
            "Kidneys and urinary tract",
            "Bowel and appendix",
            "Peritoneum and omentum",
            "Lymph nodes",
            "Vessels",
            "Abdominal wall and soft tissues",
            "Bones",
        ),
    }
    region_block_header = {"chest": "CHEST", "abdomen": "ABDOMEN"}

    def section_header_match(content: str, header: str):
        return re.search(rf"(?mi)^\s*{re.escape(header)}\s*:", content)

    def ordered_section_score(content: str, headers: tuple[str, ...]) -> float:
        positions = [
            match.start()
            for header in headers
            if (match := section_header_match(content, header)) is not None
        ]
        if not positions:
            return 0.0

        presence_score = len(positions) / len(headers)
        order_score = presence_score if positions == sorted(positions) else 0.0
        return 0.8 * presence_score + 0.2 * order_score

    def section_body(content: str, header: str, all_headers: tuple[str, ...]) -> str:
        match = section_header_match(content, header)
        if match is None:
            return ""

        next_positions = [
            next_match.start()
            for next_header in all_headers
            if next_header != header
            and (next_match := section_header_match(content[match.end():], next_header)) is not None
        ]
        end = match.end() + min(next_positions) if next_positions else len(content)
        return content[match.end():end].strip()

    contents = [completion[0]["content"] for completion in completions]
    anatomy_regions = _normalize_anatomy_regions(
        anatomy_region,
        len(contents),
        "structured_report_format_reward",
    )

    rewards = []
    for content, region in zip(contents, anatomy_regions):
        answer_start = content.find("<answer>")
        answer_end = content.find("</answer>")
        has_one_answer_block = (
            content.count("<answer>") == 1
            and content.count("</answer>") == 1
            and answer_start >= 0
            and answer_end > answer_start
        )
        report_text = content[:answer_start] if answer_start >= 0 else content
        if "</think>" in report_text:
            report_text = report_text.rsplit("</think>", 1)[1]

        headers = region_headers[region]
        block_header = region_block_header[region]
        findings_match = section_header_match(report_text, "FINDINGS")
        block_match = section_header_match(report_text, block_header)
        impression_match = section_header_match(report_text, "IMPRESSION")

        common_positions = [
            findings_match.start() if findings_match is not None else None,
            block_match.start() if block_match is not None else None,
            impression_match.start() if impression_match is not None else None,
            answer_start if has_one_answer_block else None,
        ]
        present_common_positions = [pos for pos in common_positions if pos is not None]
        common_presence_score = len(present_common_positions) / len(common_positions)
        common_order_score = (
            common_presence_score
            if present_common_positions == sorted(present_common_positions)
            else 0.0
        )
        common_score = 0.8 * common_presence_score + 0.2 * common_order_score

        region_score = ordered_section_score(report_text, headers)

        leaf_headers = tuple(header for header in headers if header != block_header)
        section_stop_headers = ("FINDINGS", *headers, "IMPRESSION")
        nonempty_sections = sum(
            bool(section_body(report_text, header, section_stop_headers))
            for header in leaf_headers
        )
        impression_text = section_body(
            report_text, "IMPRESSION", section_stop_headers
        )
        nonempty_score = (
            nonempty_sections + bool(impression_text)
        ) / (len(leaf_headers) + 1)
        impression_numbered = bool(
            re.search(r"(?m)^\s*(?:\d+[\.\)]|-)\s+\S", impression_text)
        )

        reward = (
            0.15 * common_score
            + 0.55 * region_score
            + 0.20 * nonempty_score
            + 0.10 * float(impression_numbered)
        )
        rewards.append(min(max(reward, 0.0), 1.0))

    return rewards


def accuracy_reward_f1(completions, solution, anatomy_region=None):
    """Region-adaptive set F1 over labels in the first answer block.

    An empty <answer></answer> intentionally means "No Chest Finding" or
    "No Abdominal Finding", depending on the region. Gold solutions must name
    that label explicitly; a missing answer block remains invalid.
    """
    contents = [completion[0]["content"] for completion in completions]
    anatomy_regions = _normalize_anatomy_regions(
        anatomy_region,
        len(contents),
        "accuracy_reward_f1",
    )
    rewards = []

    for index, (content, sol, region) in enumerate(
        zip(contents, solution, anatomy_regions)
    ):
        gold_parsed = _parse_and_validate_gold_labels(
            sol,
            region,
            index,
            "accuracy_reward_f1",
        )
        answer_text = _extract_answer_text(content)
        if answer_text is None:
            logger.debug(f"no answer match for content: {content} and sol: {sol}")
            rewards.append(0.0)
            continue

        answer_parsed = _parse_label_set(answer_text)
        if not answer_parsed:
            answer_parsed = {REGION_NO_FINDING_LABEL[region]}
        if _prediction_is_contradictory(answer_parsed, region):
            rewards.append(0.0)
            continue

        intersect = gold_parsed & answer_parsed
        denominator = len(gold_parsed) + len(answer_parsed)
        reward = (2.0 * len(intersect) / denominator) if denominator > 0 else 0.0
        logger.debug(
            f"f1_reward: {reward} anatomy_region: {region} "
            f"answer_parsed: {answer_parsed} gold_parsed: {gold_parsed} "
            f"intersect: {intersect}"
        )
        rewards.append(reward)

    return rewards


def format_reward(completions) -> list[float]:
    """Reward a parseable <answer>...</answer> block, with or without thinking text."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = [1.0 if _extract_answer_text(content) is not None else 0.0 for content in contents]

    for i, (reward, content) in enumerate(zip(rewards, contents)):
        if reward == 0.0:
            logger.debug(f"format_reward wrong {i} of {rewards} completion_contents {len(content)}: {content}")

    return rewards


def tag_count_reward(completions) -> list[float]:
    """Require exactly one answer block; no think tags are required by this format."""
    contents = [completion[0]["content"] for completion in completions]
    return [
        1.0 if content.count("<answer>") == 1 and content.count("</answer>") == 1 else 0.0
        for content in contents
    ]


def accuracy_reward_hard(completions, solution, anatomy_region=None, **kwargs):
    """Region-adaptive F1 gated by VLM3D answer-block formatting."""
    f_val = format_reward(completions)
    t_val = tag_count_reward(completions)
    c_val = [f * t for f, t in zip(f_val, t_val)]

    rewards = accuracy_reward_f1(
        completions,
        solution,
        anatomy_region=anatomy_region,
    )
    return [reward * gate for reward, gate in zip(rewards, c_val)]


def soft_band_punishment(completion_ids: list[list[int]], **kwargs) -> list[float]:
    """Penalize completions outside the report+answer target length band.

    The reward is zero inside the target band, so report lengths in that range
    are treated equally and accuracy/format rewards remain the main objective.
    Outside the band, the penalty ramps linearly down to -1: very short outputs
    are discouraged from collapsing into answer-only completions, while very
    long outputs are discouraged from padding or rambling past the useful report
    length.
    """
    min_completion_len = 180
    max_completion_len = 900
    soft_undershoot_cache = 180
    soft_overshoot_cache = 300

    rewards = []
    for ids in completion_ids:
        completion_length = len(ids)
        if completion_length < min_completion_len:
            rewards.append(max(-1, (completion_length - min_completion_len) / soft_undershoot_cache))
        elif completion_length > max_completion_len:
            rewards.append(max(-1, (max_completion_len - completion_length) / soft_overshoot_cache))
        else:
            rewards.append(0.0)

    return rewards
