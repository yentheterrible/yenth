import os
import sys
from pathlib import Path
from rlbot.setup_manager import SetupManager
from rlbot.utils.structures.start_match_structures import MatchConfiguration, PlayerConfiguration

def boot_match_instance():
    cfg_path = Path(__file__).parent / "bot.cfg"
    if not cfg_path.exists():
        sys.exit(f"[!] Configuration critical error: Missing targets at {cfg_path}")

    match_layout = MatchConfiguration()
    match_layout.game_mode = 0  
    
    agent = PlayerConfiguration()
    agent.bot = True
    agent.team = 0  
    agent.name = "yenth"
    
    target = PlayerConfiguration()
    target.bot = True
    target.team = 1  
    target.name = "Target_Dummy"
    
    match_layout.player_configs = [agent, target]
    
    core_engine = SetupManager()
    
    try:
        print(f"[*] Mapping asset targets to configuration: {cfg_path.name}")
        core_engine.load_config(config_location=str(cfg_path))
        
        print("[*] Launching game process pipeline hooks...")
        core_engine.setup_match(match_layout)
        print("[+] Thread payload successfully deployed.")
        
    except Exception as network_error:
        print(f"[!] Target validation exception: {network_error}")

if __name__ == "__main__":
    boot_match_instance()
