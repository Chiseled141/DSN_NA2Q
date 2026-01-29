"""
HiT-MAC Environment Factory.

Creates the HiT-MAC DSN environment from the bundled ENV module.
"""

from __future__ import division
import sys
import os

# Add HiT-MAC directory to path to enable imports
_hitmac_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'HiT-MAC')
if _hitmac_path not in sys.path:
    sys.path.insert(0, _hitmac_path)


def create_env(env_id, args, rank=-1):
    """Create HiT-MAC DSN environment.
    
    Args:
        env_id: Environment identifier ('Pose-v0' for executor, 'Pose-v1' for coordinator)
        args: Arguments containing render_save flag
        rank: Worker rank (unused, for compatibility)
    
    Returns:
        Initialized environment instance
    """
    if 'v0' in env_id:
        # Single-agent executor environment
        import ENV.DigitalPose2DBase as poseEnv
    else:
        # Multi-agent coordinator environment  
        import ENV.DigitalPose2D as poseEnv

    env = poseEnv.gym.make(env_id, getattr(args, 'render_save', False))

    return env

