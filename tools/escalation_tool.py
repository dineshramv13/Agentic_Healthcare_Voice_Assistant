"""
tools/escalation_tool.py

EscalationTool: used by agent/nodes/emergency.py to log that an emergency
redirect occurred. This is a pure logging/audit action — it never decides
WHETHER something is an emergency (that's safety.py's keyword check and
intent.py's classifier); it only records that the emergency path fired,
for audit and evaluation purposes (e.g. "how many emergency redirects
happened in this eval run" is a real, useful metric).

Input:  original user message + session_id
Output: structured emergency escalation record
"""

import logging
import uuid
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class EscalationTool:
    name = "log_emergency_escalation"
    description = (
        "Logs that the assistant redirected a patient to emergency services "
        "(999/112), for audit and clinical safety review."
    )

    def log_escalation(self, user_message: str, session_id: str) -> Dict:
        escalation_ref = f"ESC-{uuid.uuid4().hex[:8].upper()}"
        result = {
            "action": "emergency_escalation_logged",
            "escalation_reference": escalation_ref,
            "session_id": session_id,
            "original_message": user_message,
            "logged_at": datetime.utcnow().isoformat(),
        }
        logger.warning("EMERGENCY ESCALATION LOGGED: %s | session=%s", escalation_ref, session_id)
        return result
