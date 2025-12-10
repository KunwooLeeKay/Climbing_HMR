"""
Training Configuration for Climbing Mocap Refinement
"""

import json
from pathlib import Path


class TrainingConfig:
    """Configuration class for training parameters"""
    
    # Data paths
    DATA_ROOT = '/home/kunwoo/Linux_Folder/Ascend_Motion_Dataset/AscendMotion_Dataset_Release_v1/Dataset_Train_2D'
    GVHMR_OUTPUT_ROOT = '/home/kunwoo/Kunwoo/GVHMR/outputs/ascendmotion_merged'
    SMPL_MODEL_PATH = 'smpl_models'
    WALL_MESH_FILE = 'single_view.ply'
    WALL_REF_IMAGE = 'wall1.png'
    WALL_SEGMENTS_FILE = 'wall_mesh_segments.npy'
    
    # Training parameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 1  # Process one session at a time
    LEARNING_RATE_WALL = 1e-4
    LEARNING_RATE_SMPL = 1e-4
    
    # Loss weights
    CONTACT_WEIGHT = 1.0
    DEPTH_WEIGHT = 0.0  # Set to 1.0 when LiDAR available
    PENETRATION_WEIGHT = 10.0
    REG_WALL_WEIGHT = 0.1
    REG_SMPL_WEIGHT = 0.01
    
    # Alignment parameters
    ALIGNMENT_CONTACT_THRESHOLD = 0.2  # 20cm
    ALIGNMENT_NUM_SAMPLES = 2000
    ALIGNMENT_TRIM_FRACTION = 0.5
    ALIGNMENT_FIT_MODE = 'scale_trans'  # 'sim3', 'scale_trans', or 'scale_only'
    
    # MLP architecture
    WALL_MLP_HIDDEN_DIMS = [128, 256, 128]
    SMPL_MLP_HIDDEN_DIMS = [256, 512, 256]
    DROPOUT_RATE = 0.1
    
    # Optimization
    GRADIENT_CLIP_NORM = 1.0
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    
    # Checkpointing
    CHECKPOINT_DIR = 'checkpoints'
    SAVE_EVERY_N_EPOCHS = 10
    
    # Device
    DEVICE = 'cuda'
    
    # Session filtering
    WALL_PREFIX_FILTER = '20240927'  # Only use Wall 1 sessions
    SESSION_NAME_FILTER = 'WJY'      # Additional filter
    
    @classmethod
    def save(cls, filepath):
        """Save configuration to JSON file"""
        config_dict = {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
        Path(filepath).write_text(json.dumps(config_dict, indent=2))
        print(f"✓ Saved config to: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load configuration from JSON file"""
        config_dict = json.loads(Path(filepath).read_text())
        for k, v in config_dict.items():
            setattr(cls, k, v)
        print(f"✓ Loaded config from: {filepath}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        
        print("\nData:")
        print(f"  Data root: {cls.DATA_ROOT}")
        print(f"  GVHMR output: {cls.GVHMR_OUTPUT_ROOT}")
        
        print("\nTraining:")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  LR (wall): {cls.LEARNING_RATE_WALL}")
        print(f"  LR (SMPL): {cls.LEARNING_RATE_SMPL}")
        
        print("\nLoss weights:")
        print(f"  Contact: {cls.CONTACT_WEIGHT}")
        print(f"  Depth: {cls.DEPTH_WEIGHT}")
        print(f"  Penetration: {cls.PENETRATION_WEIGHT}")
        print(f"  Reg (wall): {cls.REG_WALL_WEIGHT}")
        print(f"  Reg (SMPL): {cls.REG_SMPL_WEIGHT}")
        
        print("\nAlignment:")
        print(f"  Threshold: {cls.ALIGNMENT_CONTACT_THRESHOLD}m")
        print(f"  Samples: {cls.ALIGNMENT_NUM_SAMPLES}")
        print(f"  Trim fraction: {cls.ALIGNMENT_TRIM_FRACTION}")
        print(f"  Fit mode: {cls.ALIGNMENT_FIT_MODE}")
        
        print("\nArchitecture:")
        print(f"  Wall MLP: {cls.WALL_MLP_HIDDEN_DIMS}")
        print(f"  SMPL MLP: {cls.SMPL_MLP_HIDDEN_DIMS}")
        print(f"  Dropout: {cls.DROPOUT_RATE}")
        
        print("="*60 + "\n")


# Default configuration
config = TrainingConfig()


if __name__ == "__main__":
    # Print and save default configuration
    config.print_config()
    config.save('train_config.json')