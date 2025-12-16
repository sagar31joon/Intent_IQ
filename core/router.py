# core/router.py

import os
import importlib
from core.logger import log


class IntentRouter:
    """
    Maps predicted intent → corresponding skill module.
    Each skill file MUST contain a function: run(text)
    """

    SKILLS_DIR = "skills"

    def __init__(self):
        self.skill_map = self._discover_skills()

    # -------------------------------------------------------------
    # Discover skill modules dynamically
    # -------------------------------------------------------------
    def _discover_skills(self):
        skills = {}
        base_path = self.SKILLS_DIR

        try:
            for file in os.listdir(base_path):
                if file.endswith(".py") and file != "__init__.py":
                    intent_name = file.replace(".py", "")
                    module_path = f"{self.SKILLS_DIR}.{intent_name}"
                    skills[intent_name] = module_path

            log.info(f"[Router] Skills discovered: {list(skills.keys())}")

        except Exception as e:
            log.error(f"[Router] Failed to scan skills directory: {e}")

        return skills

    # -------------------------------------------------------------
    # Route intent → correct skill module
    # -------------------------------------------------------------
    def route(self, intent: str, text: str):
        intent = intent.strip()

        # ---------------------------------------------------------
        # AUTO-CREATE PLACEHOLDER SKILL IF MISSING
        # ---------------------------------------------------------
        if intent not in self.skill_map:
            print(f"[Router] No existing skill for '{intent}'. Creating placeholder...")

            skill_path = os.path.join(self.SKILLS_DIR, f"{intent}.py")

            # Auto-create new skill file
            with open(skill_path, "w") as f:
                f.write(
                    "def run(text):\n"
                    f"    print('Placeholder skill executed for intent: {intent}. Input:', text)\n"
                )

            # Register new module
            self.skill_map[intent] = f"{self.SKILLS_DIR}.{intent}"
            print(f"[Router] Auto-created placeholder skill for '{intent}'")

        # ---------------------------------------------------------
        # IMPORT MODULE
        # ---------------------------------------------------------
        module_name = self.skill_map[intent]

        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            msg = f"[Router] Failed to import module '{module_name}': {e}"
            if self.logger:
                self.logger.error(msg)
            else:
                print(msg)
            return

        # ---------------------------------------------------------
        # CHECK IF run() EXISTS
        # ---------------------------------------------------------
        if not hasattr(module, "run"):
            msg = f"[Router] Skill '{intent}' has no run(text) function."
            if self.logger:
                self.logger.error(msg)
            else:
                print(msg)
            return

        # ---------------------------------------------------------
        # EXECUTE SKILL
        # ---------------------------------------------------------
        try:
            return module.run(text)
        except Exception as e:
            msg = f"[Router] Error running skill '{intent}': {e}"
            if self.logger:
                self.logger.error(msg)
            else:
                print(msg)
            return