"""
tools/callback_tool.py

CallbackTool: used by the fallback node (and optionally the generator)
when the system cannot confidently answer and a human callback is the
right escalation path — e.g. repeated verification failures, or a request
that's clearly out of the assistant's competence but not an emergency
(complex clinical judgement calls, complaints, sensitive personal matters).

Input:  reason for callback + session_id
Output: structured callback request record (logged, not actually dialed —
        no telephony integration in this local build)
"""

import logging
import uuid
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class CallbackTool:
    name = "request_callback"
    description = (
        "Logs a request for a human receptionist or clinician to call the "
        "patient back, for situations the AI assistant cannot safely resolve."
    )

    def request_callback(self, reason: str, session_id: str) -> Dict:
        callback_ref = f"CB-{uuid.uuid4().hex[:8].upper()}"
        result = {
            "action": "callback_requested",
            "callback_reference": callback_ref,
            "session_id": session_id,
            "reason": reason,
            "requested_at": datetime.utcnow().isoformat(),
        }
        logger.info("CallbackTool logged a callback request: %s (reason: %s)", callback_ref, reason)
        return result
