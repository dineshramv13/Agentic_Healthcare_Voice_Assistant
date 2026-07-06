"""
tools/appointment_tool.py

AppointmentTool: handles the structured/actionable side of the "appointment"
intent — distinct from FAQTool, which only answers informational questions
about appointment *policy*. This tool represents where real booking-system
integration would plug in (e.g. calling an NHS booking API or a practice
management system's API).

For this local, zero-cost project, booking against a real calendar system
is out of scope — this tool returns a structured, deterministic response
that simulates what a real booking action would confirm, and logs the
"booking intent" so it's visible in traces/evaluation. The generator node
still grounds its final reply in FAQTool's retrieved policy chunks for
anything informational (e.g. "same-day slots release at 8am").

Input:  patient message + extracted slot info (best-effort, simple)
Output: a structured dict describing the simulated action taken

"""

import logging
import uuid
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class AppointmentTool:
    name = "appointment_action"
    description = (
        "Simulates booking, rescheduling, or cancelling a GP appointment. "
        "In production this would call the practice's real scheduling system."
    )

    def simulate_booking(self, patient_message: str, session_id: str) -> Dict:
        """
        Simulates a booking action. Does not call any real calendar —
        returns a deterministic confirmation object so the generator node
        has something concrete to reference, and so this action is visible
        in the trace log / eval transcripts.
        """
        booking_ref = f"APT-{uuid.uuid4().hex[:8].upper()}"
        result = {
            "action": "appointment_request_logged",
            "booking_reference": booking_ref,
            "session_id": session_id,
            "logged_at": datetime.utcnow().isoformat(),
            "note": (
                "This is a simulated booking action (no real calendar system "
                "connected in this local build). A receptionist would confirm "
                "the exact slot by phone or the online portal."
            ),
        }
        logger.info("AppointmentTool simulated booking action: %s", booking_ref)
        return result
