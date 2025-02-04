import torch

GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM_CHECKPOINT_PATH  = "src\models\sam_vit_b_01ec64.pth"  # Ajusta a la ruta local del checkpoint si es necesario
SAM_MODEL_TYPE       = "vit_b"                # "vit_h", "vit_l", "vit_b", etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"