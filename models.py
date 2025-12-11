# models.py
from dataclasses import dataclass

@dataclass
class PromptData:
    role: str = ""
    context: str = ""
    expectations: str = ""

    def combined(self) -> str:
        """
        Combine role, context and expectations into a single prompt.
        Mirrors the original combine_prompt() behaviour.
        """
        parts = []
        role_clean = self.role.strip()
        ctx_clean = self.context.strip()
        exp_clean = self.expectations.strip()

        if role_clean:
            parts.append(f"Role: {role_clean}")
        if ctx_clean:
            parts.append(f"Context: {ctx_clean}")
        if exp_clean:
            parts.append(f"Expectations: {exp_clean}")
            
        # print(f"Combined prompt parts: {parts}")  # Debug print

        return "\n\n".join(parts)
