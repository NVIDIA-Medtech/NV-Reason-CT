import logging
import os

from transformers import TrainerCallback


logger = logging.getLogger(__name__)


def write_per_modality_processor_configs(processor, save_dir: str) -> None:
    """Save each sub-processor under `save_dir`:
      * `<save_dir>/preprocessor_config.json`        -> stock 2D image processor
      * `<save_dir>/video_preprocessor_config.json`  -> stock video processor
      * `<save_dir>/image_processor_3d/preprocessor_config.json` -> our 3D loader

    No-op if `save_dir` doesn't exist or the processor is missing any of
    the three sub-processors.
    """
    if not os.path.isdir(save_dir):
        logger.warning(f"[vlm3d-callback] save_dir does not exist, skipping: {save_dir}")
        return

    image_processor = getattr(processor, "image_processor", None)
    video_processor = getattr(processor, "video_processor", None)
    image_processor_3d = getattr(processor, "image_processor_3d", None)

    if image_processor is not None:
        image_processor.save_pretrained(save_dir)
    if video_processor is not None:
        video_processor.save_pretrained(save_dir)
    if image_processor_3d is not None:
        image_processor_3d.save_pretrained(os.path.join(save_dir, "image_processor_3d"))


class PerModalityProcessorSaveCallback(TrainerCallback):
    """Writes per-sub-processor configs into every intermediate
    `checkpoint-{step}` directory created during training.

    Constructed with a reference to the processor (so we don't have to dig
    through `trainer.processing_class` from inside the callback). Only the
    main process writes (per `on_save` lifecycle convention).
    """

    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        # `on_save` fires after each periodic checkpoint save. Compute the
        # checkpoint dir from state.global_step and write the per-modality
        # files into it.
        if not state.is_world_process_zero:
            return control
        if state.global_step is None:
            return control
        # Trainer uses PREFIX_CHECKPOINT_DIR = "checkpoint" by convention.
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            # The directory may have been renamed by save_total_limit.
            return control
        write_per_modality_processor_configs(self.processor, checkpoint_dir)
        logger.info(
            f"[vlm3d-callback] wrote per-modality preprocessor configs into {checkpoint_dir}"
        )
        return control
